# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import time

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher

from .deformable_transformer import build_deforamble_transformer
from pdvc.CaptioningHead import build_captioner
import copy
from .criterion import SetCriterion, ContrastiveCriterion
from .rl_tool import init_scorer, get_caption_reward
from misc.utils import decide_two_stage
from .base_encoder import build_base_encoder
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np
from itertools import chain

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PDVC(nn.Module):
    """ This is the PDVC module that performs dense video captioning """

    def __init__(self, base_encoder, text_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.enable_contrastive = opt.enable_contrastive
        self.enable_layer_diff_text_feature = opt.enable_layer_diff_text_feature
        self.base_encoder = base_encoder
        num_pred_text = 0
        if self.enable_contrastive or text_encoder is not None:
            self.text_encoder = text_encoder
            num_pred_text += 1
        self.enable_word_context_modeling = opt.enable_word_context_modeling
        self.enable_sentence_context_modeling = opt.enable_sentence_context_modeling
        self.transformer = transformer
        self.caption_head = captioner

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        support_mlp_class_head = vars(opt).get('support_mlp_class_head', False)
        if support_mlp_class_head:
            self.class_head = MLP(hidden_dim, hidden_dim, num_classes, 3)
        else:
            self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)

        # Text encoder
        if self.enable_contrastive:
            text_encoder_hidden_dim = self.text_encoder.config.hidden_size

            if self.opt.enable_sentence_context_modeling:
                self.sentence_context_model = SentenceLevelContextModeling(
                    opt=self.opt, hidden_dim=text_encoder_hidden_dim
                )
                num_pred_text += 1
            else:
                self.sentence_context_model = None

            if self.enable_word_context_modeling:
                if self.opt.word_context_modeling_type == 'attention_pool':
                    self.word_context_model = WordLevelContextModeling(text_encoder_hidden_dim)
                elif self.opt.word_context_modeling_type == 'max_pool':
                    self.word_context_model = lambda x,y: torch.max(x, dim=1)[0]
                elif self.opt.word_context_modeling_type == 'mean_pool':
                    self.word_context_model = lambda x,y: (x * y.unsqueeze(2)).sum(dim=1) / (1e-5 + y.sum(dim=1, keepdim=True))
            else:
                self.word_context_model = None

            # Contrastive projection
            if self.opt.enable_multilayer_projection:
                self.contrastive_projection_event = MLP(hidden_dim, hidden_dim, opt.contrastive_hidden_size, 2)
                self.contrastive_projection_text = MLP(text_encoder_hidden_dim, text_encoder_hidden_dim, opt.contrastive_hidden_size, 2)
            else:
                self.contrastive_projection_event = nn.Linear(hidden_dim, opt.contrastive_hidden_size)
                self.contrastive_projection_text = nn.Linear(text_encoder_hidden_dim, opt.contrastive_hidden_size)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if not support_mlp_class_head:
            self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)

        box_head_init_bias = vars(opt).get('box_head_init_bias', -2.0)        

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], box_head_init_bias)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], box_head_init_bias)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        if self.enable_contrastive:
            if opt.disable_cl_proj_layer_share_weight:
                self.contrastive_projection_event = _get_clones(self.contrastive_projection_event, num_pred)
                self.contrastive_projection_text = _get_clones(self.contrastive_projection_text, num_pred_text)
            else:
                self.contrastive_projection_event = nn.ModuleList(
                    [self.contrastive_projection_event for _ in range(num_pred)])
                self.contrastive_projection_text = nn.ModuleList(
                    [self.contrastive_projection_text for _ in range(num_pred_text)])

        self.translator = translator
        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')
        if opt.enable_e2t_cl:
            self.background_embed = nn.Parameter(torch.randn(1, opt.contrastive_hidden_size), requires_grad=True)
        else:
            self.background_embed = None
        if self.opt.caption_loss_type == 'rl':
            self.scorers = init_scorer(opt.rl_scorer_types, opt.cached_tokens)
            self.scorer_weights = {k:v for k,v in zip(opt.rl_scorer_types, opt.rl_scorer_weights)}

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def class_head_paramenters(self):
        class_head_params = []
        for name, value in self.named_parameters():
            if 'class_head' in name and value.requires_grad:
                class_head_params.append(value)
        return class_head_params

    def captioner_parameters(self):
        caption_params = []
        for name, value in self.named_parameters():
            if 'caption_head' in name and value.requires_grad:
                caption_params.append(value)
        return caption_params

    def bbox_head_parameters(self):
        bbox_head_params = []
        for name, value in self.named_parameters():
            if 'bbox_head' in name and value.requires_grad:
                bbox_head_params.append(value)
        return bbox_head_params

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras

    def text_encoding(self, text_encoder_input, gt_cap_num, memory=None, enable_sentence_context_modeling=False):
        dtype = next(self.parameters()).dtype
        enable_grad = self.opt.text_encoder_learning_strategy != 'frozen' and self.training
        use_amp = self.opt.train_use_amp if self.training else self.opt.eval_use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.set_grad_enabled(enable_grad):
                text_embed = self.text_encoder(**text_encoder_input)
        word_embed = text_embed.last_hidden_state.to(dtype=dtype)
        text_embed = self.word_context_model(word_embed, text_encoder_input[
            'attention_mask']) if self.enable_word_context_modeling else text_embed['pooler_output'].to(dtype=dtype)
        text_feature = []
        pre_proj_text_feat = []
        pre_proj_aux_text_feat = []
        aux_text_feature = []
        cross_attention_scores = []
        base = 0
        for i, cap_num in enumerate(gt_cap_num):
            text_feat = text_embed[base: base + cap_num]
            base += cap_num
            if self.enable_layer_diff_text_feature:
                if self.opt.word_context_modeling_type == 'attention_pool':
                    aux_text_feat = text_feat
                else:
                    aux_text_feat = F.gelu(text_feat)
                aux_text_feat_cl = self.contrastive_projection_text[0](aux_text_feat)
                aux_text_feature.append(aux_text_feat_cl)
                pre_proj_aux_text_feat.append(aux_text_feat)
            if self.enable_sentence_context_modeling:
                text_feat, cross_att_score = self.sentence_context_model(text_feat, memory[i])
                cross_attention_scores.append(cross_att_score)
            cl_text_feat = self.contrastive_projection_text[-1](text_feat)
            pre_proj_text_feat.append(text_feat)
            text_feature.append(cl_text_feat)
        aux_text_feature = aux_text_feature if self.enable_layer_diff_text_feature else text_feature
        pre_proj_aux_text_feature = pre_proj_aux_text_feat if self.enable_layer_diff_text_feature else pre_proj_text_feat
        text_embed = [aux_text_feature, text_feature]
        pre_proj_embed = [pre_proj_aux_text_feature, pre_proj_text_feat]
        return text_embed, word_embed, cross_attention_scores, pre_proj_embed

    def forward(self, dt, criterion, contrastive_criterion, transformer_input_type, eval_mode=False):
        vf = dt['video_tensor']  # (N, L, C)
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        text_encoder_input = dt['text_encoder_input'] if self.enable_contrastive else False
        gt_cap_num = [len(sents) for sents in dt['cap_raw']]
        N, L, C = vf.shape

        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                  lvl_pos_embed_flatten, mask_flatten)

        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                dt, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(proposals)
        else:
            query_embed = self.query_embed.weight
            proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory, query_embed)
        # (layer, bsz, max_event_num, feat_dim)
        hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embed,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)                                                        

        if self.enable_contrastive and text_encoder_input:
            text_embed, _, _, pre_proj_text_feat = self.text_encoding(text_encoder_input, gt_cap_num, memory)
        else:
            text_embed = None
            pre_proj_text_feat = None

        # project to co-embedding space
        if self.enable_contrastive:
            event_embed = torch.stack([self.contrastive_projection_event[i](hs_i) for i, hs_i in enumerate(hs)])
        else:
            event_embed = hs

        others = {'memory': memory,
                  'mask_flatten': mask_flatten,
                  'spatial_shapes': temporal_shapes,
                  'level_start_index': level_start_index,
                  'valid_ratios': valid_ratios,
                  'proposals_mask': proposals_mask,
                  'text_embed': text_embed,
                  'event_embed': event_embed,
                  'pre_proj_text_embed': pre_proj_text_feat}

        if eval_mode or self.opt.caption_loss_coef == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, contrastive_criterion, hs, query_embed,
                                                      init_reference, inter_references, others,
                                                      disable_iterative_refine, self.opt.eval_disable_captioning)
        else:
            if self.opt.set_cost_caption > 0:
                out, loss = self.parallel_prediction_full_train(dt, criterion, contrastive_criterion, hs, query_embed,
                                                             init_reference, inter_references, others,
                                                             disable_iterative_refine)
            else:
                out, loss = self.parallel_prediction_matched(dt, criterion, contrastive_criterion, hs, query_embed, init_reference, inter_references, others,
                                                         disable_iterative_refine)

        return out, loss

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0
    
    
    def parallel_prediction_full_train(self, dt, criterion, contrastive_criterion, hs, query_embed, init_reference, inter_references, others,
                                 disable_iterative_refine, disable_captioning=False):
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []
        cap_cost_mats = []

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            hs_for_cap = torch.cat([hs_lid, query_embed], dim=-1) if vars(self.opt).get('enable_pos_emb_for_captioner', False) else hs_lid
            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            # if l_id != hs.shape[0] - 1 or disable_captioning:
            if disable_captioning:
                # cap_probs, seq = self.caption_prediction(
                #     self.caption_head[l_id], dt, hs_for_cap, reference, others, 'none')
                cap_loss, cap_probs, seq, cost_cap = self.caption_prediction(
                    self.caption_head[l_id], dt, hs_for_cap, reference, others, 'none')
            else:
                # cap_probs, seq = self.caption_prediction(
                #     self.caption_head[l_id], dt, hs_for_cap, reference, others, self.opt.caption_decoder_type)
                cap_loss, cap_probs, seq, cost_cap = self.caption_prediction(
                    self.caption_head[l_id], dt, hs_for_cap, reference, others, self.opt.caption_decoder_type)
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            if self.enable_contrastive:
                if len(others['text_embed']) < num_pred:
                    raw_text_emd, context_text_embed = others['text_embed']
                    text_embed_new = [raw_text_emd] * (num_pred-1) + [context_text_embed]
                    others['text_embed'] = text_embed_new
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))

                text_embed = torch.cat(others['text_embed'][l_id], dim=0)
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                # pdb.set_trace()
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
            cap_cost_mats.append(cost_cap)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        cap_cost_mat = torch.stack(cap_cost_mats)  # [decoder_layer, bs, num_query, 4]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs,
                   'cl_match_mats': cl_match_mats,
                   'cap_cost_mat': cap_cost_mat}
        if others['event_embed'] is not None:
            all_out['event_embed'] = others['event_embed']
        if others['text_embed'] is not None:
            all_out['text_embed'] = others['text_embed']

        out = {k: v[-1] for k, v in all_out.items()}
        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
        out['memory'] = others['memory'] # used for generating text features of genrated captions
        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        if self.enable_contrastive:
            for l_id in range(hs.shape[0]):
                if not self.aux_loss and l_id == 0:
                    continue
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                contrastive_loss, logits = contrastive_criterion(
                    text_embed=others['text_embed'][l_id],
                    event_embed=others['event_embed'][l_id],
                    matching_indices=indices,
                    return_logits=True,
                    bg_embed = self.background_embed,
                )
                out['cl_logits'] = logits
                l_dict = {'contrastive_loss': contrastive_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)
        return out, loss

    def parallel_prediction_full(self, dt, criterion, contrastive_criterion, hs, query_embed, init_reference,
                                 inter_references, others,
                                 disable_iterative_refine, disable_captioning=False):
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            hs_for_cap = torch.cat([hs_lid, query_embed], dim=-1) if vars(self.opt).get('enable_pos_emb_for_captioner', False) else hs_lid
            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1 or disable_captioning:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_for_cap, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_for_cap, reference, others, self.opt.caption_decoder_type)

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            if self.enable_contrastive:
                if len(others['text_embed']) < num_pred:
                    raw_text_emd, context_text_embed = others['text_embed']
                    text_embed_new = [raw_text_emd] * (num_pred - 1) + [context_text_embed]
                    others['text_embed'] = text_embed_new
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))

                text_embed = torch.cat(others['text_embed'][l_id], dim=0)
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                # pdb.set_trace()
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs,
                   'cl_match_mats': cl_match_mats}
        if others['event_embed'] is not None:
            all_out['event_embed'] = others['event_embed']
        if others['text_embed'] is not None:
            all_out['text_embed'] = others['text_embed']
        all_out['event_feat'] = hs

        out = {k: v[-1] for k, v in all_out.items()}
        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        if self.enable_contrastive:
            for l_id in range(hs.shape[0]):
                if not self.aux_loss and l_id == 0:
                    continue
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                contrastive_loss, logits = contrastive_criterion(
                    text_embed=others['text_embed'][l_id],
                    event_embed=others['event_embed'][l_id],
                    matching_indices=indices,
                    return_logits=True,
                    bg_embed = self.background_embed,
                )
                out['cl_logits'] = logits
                l_dict = {'contrastive_loss': contrastive_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

        return out, loss

    def parallel_prediction_matched(self, dt, criterion, contrastive_criterion, hs, query_embed, init_reference,
                                    inter_references, others,
                                    disable_iterative_refine):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            hs_for_cap = torch.cat([hs_lid, query_embed], dim=-1) if vars(self.opt).get('enable_pos_emb_for_captioner', False) else hs_lid
            loss_caption, cap_probs, seq, cost_cap = self.caption_prediction(self.caption_head[l_id], dt, hs_for_cap,
                                                                                 reference, others, 'none')
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            if self.enable_contrastive and self.opt.set_cost_cl > 0:
                if len(others['text_embed']) < num_pred:
                    raw_text_emd, context_text_embed = others['text_embed']
                    text_embed_new = [raw_text_emd] * (num_pred - 1) + [context_text_embed]
                    others['text_embed'] = text_embed_new
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))
                text_embed = torch.cat(others['text_embed'][l_id], dim=0)
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs,
            'cl_match_mats': cl_match_mats}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])

            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                hs_for_cap = torch.cat([hs_lid, query_embed], dim=-1) if vars(self.opt).get('enable_pos_emb_for_captioner', False) else hs_lid
                cap_loss, cap_probs, seq, cost_cap = self.caption_prediction(self.caption_head[l_id],
                                                                    dt,
                                                                    hs_for_cap, reference,
                                                                    others,
                                                                    self.opt.caption_decoder_type,
                                                                    indices)

                l_dict = {'loss_caption': cap_loss}
                if self.enable_contrastive:
                    contrastive_loss = contrastive_criterion(
                        text_embed = others['text_embed'][l_id],
                        event_embed = others['event_embed'][l_id],
                        matching_indices = indices,
                        bg_embed = self.background_embed,
                    )

                    l_dict.update({'contrastive_loss': contrastive_loss})
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

            out.update({'caption_probs': cap_probs, 'seq': seq})
        else:
            loss, last_indices = criterion(out, dt['video_target'])

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]

            hs_for_cap = torch.cat([hs_lid, query_embed], dim=-1) if vars(self.opt).get('enable_pos_emb_for_captioner', False) else hs_lid
            cap_loss, cap_probs, seq, cost_cap = self.caption_prediction(self.caption_head[l_id], dt,
                                                                                            hs_for_cap, reference,
                                                                                            others,
                                                                                            self.opt.caption_decoder_type,
                                                                                            indices)
            l_dict = {'loss_caption': cap_loss}
            if self.enable_contrastive:
                contrastive_loss = contrastive_criterion(
                    text_embed = others['text_embed'][l_id],
                    event_embed = others['event_embed'][l_id],
                    matching_indices = indices
                )

                l_dict.update({'contrastive_loss': contrastive_loss})
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})

        return out, loss

    def caption_prediction(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
        N_, N_q, C = hs.shape
        all_cap_num = len(dt['cap_tensor'])
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()

        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx = row_idx + query_nums[i]
                col_idx = col_idx + gt_nums[i]

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])

            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt['cap_tensor']
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, :len(feat_ids)] = hs[i, feat_ids]
                new_seq_for_dsa[i, :len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        cap_probs = {}
        flag = True

        if captioner_type == 'none':
            cost_caption = torch.zeros(N_, N_q, all_cap_num,
                                       device=hs.device)  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq

        elif captioner_type in ['light']:
            clip = hs_r.unsqueeze(1)
            text_embed = torch.cat(text_embed, dim=0).unsqueeze(1) if text_embed is not None else None
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None

        elif self.opt.caption_decoder_type in ['gpt2']:
            if self.opt.caption_loss_type == 'rl':
                raise NotImplementedError()
            prefix = hs_r[feat_bigids]
            cap_raw = dt['cap_raw']
            cap_loss, cap_prob = cap_head(prefix, cap_raw)
            seq = dt['cap_tensor'][cap_bigids]
            cost_cap = cap_loss.reshape(N_ * N_q, -1)
            return cap_loss.mean(), cap_probs, seq, cost_cap

        elif self.opt.caption_decoder_type in ['standard', 'transformer']:
            seq = dt['cap_tensor'][cap_bigids]
            seq_mask = dt['cap_mask'][cap_bigids]
            max_match_feat_num = max([len(feat_ids) for feat_ids, _ in indices])
            hs_for_cap_head = hs.new_zeros(N_, max_match_feat_num, C)
            query_mask_for_cap_head = hs.new_zeros(N_, max_match_feat_num)
            reference_for_cap_head = hs.new_zeros(N_, max_match_feat_num, reference.shape[-1])
            seq_for_cap_head = hs.new_zeros(N_, max_match_feat_num, seq.shape[-1], dtype=torch.long)
            seq_mask_for_cap_head = hs.new_zeros(N_, max_match_feat_num, seq.shape[-1], dtype=torch.bool)

            for i, (feat_ids, _) in enumerate(indices):
                hs_for_cap_head[i, :len(feat_ids)] = hs[i, feat_ids]
                query_mask_for_cap_head[i, :len(feat_ids)] = 1
                reference_for_cap_head[i, :len(feat_ids)] = reference[i, feat_ids]
                vid_mask = dt['gt_gather_idx'][cap_bigids] == i
                seq_new = seq[vid_mask]
                seq_mask_new = seq_mask[vid_mask]
                seq_for_cap_head[i, :len(seq_new)] = seq_new
                seq_mask_for_cap_head[i, :len(seq_mask_new)] = seq_mask_new
            seq_for_cap_head = seq_for_cap_head.flatten(0, 1)
            seq_mask_for_cap_head = seq_mask_for_cap_head.flatten(0, 1)
            if self.training:
                if self.opt.caption_loss_type != 'rl':
                    cap_prob = cap_head(hs_for_cap_head, reference_for_cap_head, others, seq_for_cap_head)
                elif self.opt.caption_loss_type == 'rl':
                    gen_result, sample_logprobs = cap_head.sample(hs_for_cap_head, reference_for_cap_head, others,
                                                                            opt={'sample_max': 0})
                    cap_head.eval()
                    with torch.no_grad():
                        greedy_res, _ = cap_head.sample(hs_for_cap_head, reference_for_cap_head, others)
                    cap_head.train()
                    gen_result = gen_result.reshape(-1, gen_result.shape[-1])
                    greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])

                    # gt_caption = [[self.translator.translate(cap, max_len=50) for cap in caps] for caps in dt['cap_raw']]
                    # gt_caption = list(chain(*gt_caption))
                    # gt_caption = [gt_caption[i] for i in cap_bigids]

                    gt_caption = seq_for_cap_head.detach().cpu().numpy()
                    if self.opt.cl_para_ratio > 0:
                        gt_para = [self.translator.translate(' '.join(caps), max_len=150) for caps in dt['cap_raw']]
                        # gt_caption = [gt_caption[cap_vid_ids[i]][cap_event_ids[i]] for i in range(len(cap_vid_ids))]
                        para_gen_result = gen_result.reshape(N_, max_match_feat_num, -1)
                        para_greedy_res = greedy_res.reshape(N_, max_match_feat_num, -1)
                        para_gen_result_new = torch.zeros((N_, max_match_feat_num, para_gen_result.size(-1)), device=para_gen_result.device, dtype=torch.int)
                        para_greedy_res_new = torch.zeros((N_, max_match_feat_num, para_greedy_res.size(-1)), device=para_greedy_res.device, dtype=torch.int)
                        for i, (feat_ids, cap_ids) in enumerate(indices):
                            cap_ids_reverse = np.argsort(cap_ids)
                            para_gen_result_new[i][:len(cap_ids_reverse)] = para_gen_result[i][:len(cap_ids_reverse)][cap_ids_reverse]
                            para_greedy_res_new[i][:len(cap_ids_reverse)] = para_greedy_res[i][:len(cap_ids_reverse)][cap_ids_reverse]
                        para_reward, para_sample_meteor, para_greedy_meteor = get_caption_reward(self.scorers, para_greedy_res_new,
                                                                                  gt_para,
                                                                                  para_gen_result_new, self.scorer_weights,
                                                                                  is_para=True)
                        para_reward = para_reward[..., None].repeat(max_match_feat_num, 1).reshape(-1)
                    else:
                        para_reward = 0
                    if self.opt.cl_sent_ratio > 0:
                        reward, sample_meteor, greedy_meteor = get_caption_reward(self.scorers, greedy_res,
                                                                                  gt_caption,
                                                                                  gen_result, self.scorer_weights)
                    else:
                        reward = 0
                    reward = para_reward * self.opt.cl_para_ratio + reward * self.opt.cl_sent_ratio
                    reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)
                    gen_result = gen_result * query_mask_for_cap_head.int().reshape(-1, 1)
                    cap_rl_loss = cap_head.build_rl_loss(sample_logprobs, gen_result.float(),
                                                                   sample_logprobs.new_tensor(reward))
                    cap_loss = cap_rl_loss
                    # cap_cost = -hs_r.new_tensor(greedy_meteor)
                    return cap_loss.mean(), cap_probs, seq, 0 * cap_loss.mean() 
            else:
                with torch.no_grad():
                    cap_prob = cap_head(hs_for_cap_head, reference_for_cap_head, others, seq_for_cap_head)
                    seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

            flag = False
            pass

        if flag:
            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_loss_type != 'rl':
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs['cap_prob_train'] = cap_prob
                elif self.opt.caption_loss_type == 'rl':
                    # gen_result: (eseq_num, eseq_len, ~cap_len), sample_logprobs :(eseq_num, eseq_len, ~cap_len)
                    gen_result, sample_logprobs = cap_head.sample(event, clip_ext, clip_mask_ext,
                                                                            opt={'sample_max': 0})
                    cap_head.eval()
                    with torch.no_grad():
                        greedy_res, _ = cap_head.sample(event, clip_ext, clip_mask_ext)
                    cap_head.train()
                    gen_result = gen_result.reshape(-1, gen_result.shape[-1])
                    greedy_res = greedy_res.reshape(-1, greedy_res.shape[-1])

                    gt_caption = [[self.translator.translate(cap, max_len=50) for cap in caps] for caps in
                                  dt['cap_raw']]
                    gt_caption = list(chain(*gt_caption))
                    # gt_caption = [gt_caption[cap_vid_ids[i]][cap_event_ids[i]] for i in range(len(cap_vid_ids))]
                    gt_caption = [gt_caption[i] for i in cap_bigids]
                    reward, sample_meteor, greedy_meteor = get_caption_reward(self.scorers, greedy_res,
                                                                              gt_caption,
                                                                              gen_result, self.scorer_weights)
                    reward = np.repeat(reward[:, np.newaxis], gen_result.size(1), 1)
                    cap_rl_loss = cap_head.build_rl_loss(sample_logprobs, gen_result.float(),
                                                                   sample_logprobs.new_tensor(reward))
                    cap_loss = cap_rl_loss
                    # cap_cost = -hs_r.new_tensor(greedy_meteor)
                    cost_cap = cap_loss.reshape(N_ * N_q, -1)
                    return cap_loss.mean(), cap_probs, seq, cost_cap
            else:
                with torch.no_grad():
                    seq_gt = dt['cap_tensor'][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

        cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
        if flag:
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]
        else:
            caption_tensor = seq_for_cap_head[:, 1:]
            caption_mask = seq_mask_for_cap_head[:, 1:]

        assert self.opt.caption_cost_type == 'loss'
        cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask)
        if self.opt.set_cost_caption > 0:
            cost_cap = cap_loss.reshape(N_ * N_q, -1)
        else:
            cost_cap = cap_loss
        return cap_loss.mean(), cap_probs, seq, cost_cap

    def caption_prediction_eval(self, cap_head, dt, hs, reference, others, decoder_type, indices=None):
        assert indices == None
        N_, N_q, C = hs.shape
        query_mask = others['proposals_mask']
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        row_idx, col_idx = 0, 0

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval

        elif decoder_type in ['gpt2']:
            prefix = hs_r
            use_amp = self.opt.train_use_amp if self.training else self.opt.eval_use_amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                gen_caps, cap_prob_eval, gen_mask = cap_head.sample(prefix, entry_length=self.opt.max_caption_len)
            cap_probs['cap_prob_eval'] = cap_prob_eval.reshape(N_, N_q, -1)
            cap_probs['gpt2_cap'] = gen_caps
            cap_probs['gen_mask'] = gen_mask.reshape(N_, N_q, -1)
            seq = None

        elif decoder_type in ['standard', 'transformer']:
            with torch.no_grad():
                seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                if len(seq):
                    seq = seq.reshape(-1, N_q, seq.shape[-1])
                    cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                cap_probs['cap_prob_eval'] = cap_prob_eval

        return cap_probs, seq


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.enable_contrastive and vars(opt).get('eval_enable_grounding', False):
            from pdvc.matcher import HungarianMatcher
            self.grounding_matcher = HungarianMatcher(cost_class=opt.eval_set_cost_class,
                            cost_bbox=0.0,
                            cost_giou=0.0,
                            cost_alpha = opt.eval_grounding_cost_alpha,
                            cost_gamma = opt.eval_grounding_cost_gamma,
                            cost_cl= opt.eval_set_cost_cl,
                            )

    @torch.no_grad()
    def forward_grounding(self, outputs, target_sizes, targets):
        if not self.opt.enable_contrastive:
            return None, None

        for target in targets:
            target['boxes'] = target['boxes'] * 0
            target['labels'] = target['labels'] * 0

        all_boxes = box_ops.box_cl_to_xy(outputs['pred_boxes'])
        all_boxes[all_boxes < 0] = 0
        all_boxes[all_boxes > 1] = 1
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        all_boxes = all_boxes * scale_fct[:, None, :]
        all_boxes = all_boxes.cpu().numpy().tolist()

        all_logits = outputs['pred_logits'].sigmoid().cpu().numpy().tolist()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        last_indices,_,C = self.grounding_matcher(outputs_without_aux, targets, return_C=True)

        def get_results(indices, C):
            results = []
            for i, (event_ind, cap_ind) in enumerate(indices):
                N_cap = len(targets[i]['boxes'])
                boxes = []
                confs = []
                cl_scores = []
                cap_ind = cap_ind.numpy().tolist()
                for j in range(N_cap):
                    if self.opt.eval_enable_maximum_matching_for_grounding:
                        event_j = C[i][:, j].argmin()
                    else:
                        if j not in cap_ind:
                            # print(C[0].shape, len(C), j)
                            event_j = C[i][:, j].argmin()
                        else:
                            match_id = cap_ind.index(j)
                            event_j = event_ind[match_id]
                    boxes.append(all_boxes[i][event_j])
                    confs.append(all_logits[i][event_j][0])
                    cl_scores.append(C[i][event_j, j].item())
                results.append({'boxes': boxes, 'confs': confs, 'cl_scores': cl_scores})
            return results

        last_results = get_results(last_indices, C)
        cl_scores = outputs['cl_match_mats']
        sizes = [len(v["boxes"]) for v in targets]
        if cl_scores.shape[1] > sum(sizes):
            bs, num_queries, _ = outputs['pred_boxes'].shape
            bg_cl_score = cl_scores[:, -1:].reshape(bs, num_queries, 1)
            cl_scores = cl_scores[:, :-1].reshape(bs, num_queries, -1)
            cl_scores = [torch.cat((c[i], bg_cl_score[i]), dim=1) for i, c in enumerate(cl_scores.split(sizes, dim=-1))]
        return last_results, cl_scores

    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader, model=None, tokenizer=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)
        bs, num_queries = boxes.shape[:2]

        if seq is None and 'gpt2_cap' in outputs['caption_probs']:
            caps = outputs['caption_probs']['gpt2_cap']
            cap_idx = 0
            caps_new = []
            for batch, b in enumerate(topk_boxes):
                caps_b = []
                for q_id, idx in enumerate(b):
                    caps_b.append(caps[cap_idx])
                    cap_idx += 1
                caps_new.append(caps_b)
            caps = caps_new
            mask = outputs['caption_probs']['gen_mask']
            cap_prob = outputs['caption_probs']['cap_prob_eval']
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            if len(seq):
                mask = (seq > 0).float()
                cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
                seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
                caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
                caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
                cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            else:
                bs, num_queries = boxes.shape[:2]
                cap_scores = [[-1e5] * num_queries] * bs
                caps = [[''] * num_queries] * bs

        if self.opt.enable_contrastive and self.opt.eval_enable_matching_score:
            event_embed = outputs['event_embed']
            cap_list = list(chain(*caps))
            text_encoder_inputs = tokenizer(cap_list, return_tensors='pt', padding=True)

            text_encoder_inputs = {key: _.to(self.opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in
                                  text_encoder_inputs.items()}

            input_cap_num = [len(_) for _ in caps]
            memory = outputs.get('memory', [None] * len(input_cap_num))
            text_embed, word_embed, _, _ = model.text_encoding(text_encoder_inputs, input_cap_num, memory=memory)

            text_embed = torch.cat(text_embed[-1], dim=0) # feature of last decoder layer
            event_embed = event_embed.reshape(-1, event_embed.shape[-1])

            normalized_text_emb = F.normalize(text_embed, p=2, dim=1)
            normalized_event_emb = F.normalize(event_embed, p=2, dim=1)
            cl_logits = torch.mm(normalized_text_emb, normalized_event_emb.t())

            sizes = [num_queries] * bs
            cl_pre_logit = [torch.eq(m.split(sizes, 0)[i].argmax(dim=1), topk_indexes[i]).sum() for i, m in enumerate(cl_logits.split(sizes, 1))]
            cl_scores = [torch.gather(m.split(sizes, 0)[i], 1, topk_indexes[i].unsqueeze(1)).squeeze(1) for i, m in enumerate(cl_logits.split(sizes, 1))]
            cl_scores = [cl_score.cpu().numpy().astype('float') for cl_score in cl_scores]
        else:
            cl_scores = [[0.0] * num_queries] * bs

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'cl_scores': cls,'query_id': qid,
             'vid_duration': ts, 'pred_seq_len': sl, 'raw_idx': idx} for s, l, b, rb, c, cs, cls, qid, ts, sl, idx in
            zip(scores, labels, boxes, raw_boxes, caps, cap_scores, cl_scores, topk_boxes, target_sizes, eseq_lens, topk_indexes)]
        return results


class WordLevelContextModeling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        query = F.gelu(self.w1(x))
        alpha = self.w2(query)
        alpha = alpha * mask.unsqueeze(2) + (-10000) * (1-mask).unsqueeze(2)
        alpha = F.softmax(alpha, dim=1)
        feat = torch.matmul(alpha.permute(0, 2, 1), x)
        return feat.squeeze(1)


class SentenceLevelContextModeling(nn.Module):
    def __init__(self, opt, hidden_dim):
        super().__init__()
        self.enable_sentence_pos_embedding = opt.enable_sentence_pos_embedding
        self.enable_cross_model_fusion = opt.enable_cross_model_fusion
        self.max_pos_num = opt.max_pos_num
        self.sentence_pos_embedding_type = opt.sentence_pos_embedding_type
        if self.enable_sentence_pos_embedding:
            if self.sentence_pos_embedding_type == 'cosine':
                self.register_buffer('pos_table', self._get_sin_encoding_table(self.max_pos_num, hidden_dim))
            else:
                self.pos_table = nn.Embedding(self.max_pos_num, hidden_dim)
        self.layer_num = opt.sentence_modeling_layer_num
        if self.enable_cross_model_fusion:
            self.memory_projection = nn.Linear(opt.hidden_dim, hidden_dim)
        self.transformer_block = BertEncoder(
            BertConfig(
                num_hidden_layers=self.layer_num,
                hidden_size=hidden_dim,
                is_decoder=self.enable_cross_model_fusion,
                add_cross_attention=self.enable_cross_model_fusion
            )
        )


    def _get_sin_encoding_table(self, max_len, hidden_dim):
        ''' Sinusoid position encoding table '''
        pos_encoding_table = torch.zeros(max_len, hidden_dim).float()
        pos_encoding_table.require_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim)).exp()
        pos_encoding_table[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding_table[:, 1::2] = torch.cos(pos * div_term)
        return pos_encoding_table.unsqueeze(0)


    def forward(self, x, memory=None):
        x = x.unsqueeze(0)
        memory = self.memory_projection(memory).unsqueeze(0) if self.enable_cross_model_fusion else None
        pos_embed = x.new_zeros((x.size(1), x.size(2)))

        if self.enable_sentence_pos_embedding:
            if self.sentence_pos_embedding_type == 'cosine':
                pos_embed_ = self.pos_table[0, :x.size(1)]
                # x = x + self.pos_table[:, :x.size(1)] if self.enable_sentence_pos_embedding else x
            else:
                pos_embed_ = self.pos_table.weight[:x.size(1)]
            pos_embed[:len(pos_embed_)] = pos_embed_
            # x = x + self.pos_table.weight[:x.size(1)] if self.enable_sentence_pos_embedding else x
            x = x + pos_embed
        x = self.transformer_block(x, encoder_hidden_states=memory, output_attentions=True, output_hidden_states=True)
        x, cross_attention_scores = x.last_hidden_state, x.cross_attentions
        if cross_attention_scores is not None:
            cross_attention_scores = cross_attention_scores[0]
            cross_attention_scores = torch.mean(cross_attention_scores, dim=1)
            return x.squeeze(0), cross_attention_scores.squeeze(0)
        return x.squeeze(0), None


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    if args.enable_contrastive:
        for i in range(10):
            try:
                text_encoder = AutoModel.from_pretrained(args.pretrained_language_model, cache_dir=args.huggingface_cache_dir)
                break
            except:
                print('download error in AutoModel, retry...')
                time.sleep(1)
    else:
        text_encoder = None
    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)

    model = PDVC(
        base_encoder,
        text_encoder,
        transformer,
        captioner,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_counter': args.count_loss_coef,
                   'loss_caption': args.caption_loss_coef,
                   'contrastive_loss': args.contrastive_loss_start_coef,
                   }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)
    contrastive_criterion = ContrastiveCriterion(temperature=args.contrastive_loss_temperature,
                                                 enable_cross_video_cl=args.enable_cross_video_cl,
                                                 enable_e2t_cl = args.enable_e2t_cl,
                                                 enable_bg_for_cl = args.enable_bg_for_cl)
    contrastive_criterion.to(device)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, contrastive_criterion, postprocessors
