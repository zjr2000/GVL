import torch
import torch.nn as nn
from pdvc.ops.modules import MSDeformAttn, MSDeformAttnCap
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertConfig
import torch.nn.functional as F
import math


def _get_extended_attention_mask(attention_mask):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class TransformerDSACapLayer(nn.Module):
    def __init__(self, d_model, d_ffn=1024, dropout=0.1, n_heads=8, n_levels=4, n_points=4, activation='relu', im2col_step=500):
        super(TransformerDSACapLayer, self).__init__()
        # Self Attention
        self.self_attention = BertSelfAttention(BertConfig(
                num_attention_heads=n_heads,
                hidden_size=d_model,
                attention_probs_dropout_prob=dropout,
                is_decoder=False))

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross Attention
        self.dim_project = nn.Linear(2 * d_model, d_model)

        self.cross_attention = MSDeformAttn(d_model, n_levels, n_heads, n_points, im2col_step)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # projection. To align dimension
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


    def forward_ffn(self, hidden_states):
        hidden_states2 = self.linear2(self.dropout3(self.activation(self.linear1(hidden_states))))
        hidden_states = hidden_states + self.dropout4(hidden_states2)
        hidden_states = self.norm3(hidden_states)
        return hidden_states

    def forward(self, hidden_states, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask, attention_mask):
        # Build uni-directional causal mask
        _, query_num, _ = query.shape
        _, _, n_levels, points_dim = reference_points.shape
        _, current_max_cap_len, _ = hidden_states.shape
        # self attention + Add + Norm
        hidden_states = hidden_states + self.dropout1(self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )[0])
        hidden_states = self.norm1(hidden_states)
        input_flatten = input_flatten.unsqueeze(1).repeat(1, query_num, 1 ,1).reshape(-1, input_flatten.shape[-2], input_flatten.shape[-1])
        input_padding_mask = input_padding_mask.unsqueeze(1).repeat(1, query_num, 1).reshape(-1, input_padding_mask.shape[-1])
        reference_points = reference_points.reshape(-1, n_levels, points_dim).unsqueeze(1).repeat(1, current_max_cap_len, 1, 1)
        query = query.reshape(-1, query.shape[-1]).unsqueeze(1).repeat(1, current_max_cap_len, 1)
        hidden_states = torch.cat([hidden_states, query], -1)
        hidden_states = self.dim_project(hidden_states)
        hidden_states = hidden_states + self.dropout2(self.cross_attention(
            hidden_states, reference_points, input_flatten, input_spatial_shapes,
            input_level_start_index, input_padding_mask
        ))
        hidden_states = self.norm2(hidden_states)
        # FFN
        hidden_states = self.forward_ffn(hidden_states)
        return hidden_states


class TransformerDSACaptioner(nn.Module):
    def __init__(self, opt):
        super(TransformerDSACaptioner, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.hidden_size = opt.hidden_dim
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len

        # Multi Scale Deformable Attention args
        self.n_levels = opt.cap_num_feature_levels
        self.n_heads = opt.cap_nheads
        self.n_points = opt.cap_dec_n_points
        # word embedding
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        # pos embedding
        self.register_buffer('pos_table', self._get_sin_encoding_table(self.max_caption_len + 2, self.input_encoding_size))
        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerDSACapLayer(
                d_model=self.hidden_size,
                n_heads=self.n_heads,
                n_levels=self.n_levels,
                n_points=self.n_points,
                dropout=self.drop_prob_lm,
                activation='relu'
            ) for _ in range(self.num_layers)]
        )
        # output logits
        self.lm_dropout = nn.Dropout(self.drop_prob_lm)
        self.logits = nn.Linear(self.hidden_size, self.vocab_size + 1)


    def init_weights(self):
        pass

    def _get_sin_encoding_table(self, max_len, hidden_dim):
        ''' Sinusoid position encoding table '''
        pos_encoding_table = torch.zeros(max_len, hidden_dim).float()
        pos_encoding_table.require_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim)).exp()
        pos_encoding_table[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding_table[:, 1::2] = torch.cos(pos * div_term)
        return pos_encoding_table.unsqueeze(0)
    

    def build_loss(self, input, target, mask):
        input = input[:, :-1, :]
        one_hot = torch.nn.functional.one_hot(target, self.opt.vocab_size+1)
        max_len = input.shape[1]
        output = - (one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output


    def forward(self, query, reference, others, seq):
        """
        hs: event level representation (batch_size, query_num, hidden_size), 
            here query_num equal to the max matched feat num in current batch
        cap_tensor: gt caption tokens (batch_size * query_num, max_cap_len)
        reference: reference points
        """

        seq = seq.long()
        reference_points = reference
        input_flatten = others['memory']
        input_spatial_shapes = others['spatial_shapes']
        input_level_start_index = others['level_start_index']
        input_padding_mask = others['mask_flatten']
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] \
                                     * torch.stack([others['valid_ratios']]*2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * others['valid_ratios'][:, None, :, None]

        if self.n_levels < self.opt.num_feature_levels:
            input_spatial_shapes = input_spatial_shapes[:self.n_levels]
            input_level_start_index = input_level_start_index[:self.n_levels]
            total_input_len = torch.prod(input_spatial_shapes, dim=1).sum()
            input_flatten = input_flatten[:, :total_input_len]
            input_padding_mask = input_padding_mask[:, :total_input_len]
            reference_points = reference_points[:, :, :self.n_levels]

        seq = self.embed(seq) # bsz * query_num, cap_len, input_encoding_size
        # position embedding
        pos_embed = seq.new_zeros((seq.size(1), seq.size(2)))
        pos_embed_ = self.pos_table[0, :seq.size(1)]
        pos_embed[:len(pos_embed_)] = pos_embed_
        seq = seq + pos_embed
        bsz_mul_query_num, current_max_cap_len, _ = seq.shape
        attention_mask = seq.new_ones(bsz_mul_query_num, current_max_cap_len, current_max_cap_len).tril()
        attention_mask = _get_extended_attention_mask(attention_mask)
        for layer in self.layers:
            seq = layer(seq, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask, attention_mask)
        logprobs = F.log_softmax(self.logits(self.lm_dropout(seq)), dim=-1)
        return logprobs       


    def sample(self, hs, reference, others, opt={}):
        vid_num, query_num, _ = hs.shape
        batch_size = vid_num * query_num
        sample_max = opt.get('sample_max', 1)

        query = hs
        seq = None
        seqLogprobs = []
        seq_returned = []
        logprobs = None
        for t in range(self.max_caption_len + 1):
            if t == 0: # input <bos>
                seq = hs.data.new(batch_size, 1).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs, -1)
                it = it.long()
                seq = torch.cat([seq, it], -1)
                it = it.view(-1)
            else:
                raise NotImplementedError('Do not support scale logprobs by temperture')

            logprobs = self.forward(query, reference, others, seq)  # (bsz, current_cap_len, vocab_size)
            logprobs = logprobs[:,-1,:].unsqueeze(1) # (bsz, 1, vocab_size)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq_returned.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs)    

        if seq_returned==[] or len(seq_returned)==0:
            return [],[]
        return torch.cat([_.unsqueeze(1) for _ in seq_returned], 1), torch.cat(seqLogprobs, 1)