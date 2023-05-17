import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from tqdm import tqdm, trange
import logging
import numpy as np
from itertools import chain
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def forward(self, prefix, captions, require_labels=True):
        device = next(self.parameters()).device
        captions = list(chain(*captions))
        data = self.tokenizer(captions, padding='longest', return_tensors="pt")
        tokens = data['input_ids'].to(device)
        b, l = tokens.size()
        mask = data['attention_mask'].to(device)
        mask = torch.cat([torch.ones((b, self.prefix_length), device=device), mask], dim=1)
        outputs = self._forward(tokens, prefix, mask, labels=require_labels)
        return outputs.loss, outputs.logits[:, self.prefix_length-1: -1]

    def _forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[bool] = False):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels:
            dummy_token = torch.ones(tokens.shape[0], self.prefix_length, dtype=torch.int64, device=tokens.device)
            labels = torch.cat((-100 * dummy_token, tokens), dim=1) # -100 indicating masked tokens for loss calculation
            labels = labels * mask + (1-mask) * -100 # forbid padding tokens from calculating loss
            labels = labels.to(dtype=torch.int64)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    @torch.no_grad()
    def sample(self, embed, entry_count=1, entry_length=67, temperature=1., stop_token='.', tokens=None):
        embed = self.clip_project(embed).reshape(-1, self.prefix_length, self.gpt_embedding_size)
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        generated_list = []
        generated_mask_list =[]
        next_token_probs_list = []

        for entry_idx in range(entry_count):
            generated = embed
            generated_mask = []
            next_token_probs = []
            break_flag = torch.tensor([False]*embed.shape[0], dtype=torch.bool,device=embed.device)
            for i in range(entry_length):
                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                next_token = torch.argmax(logits, -1).unsqueeze(1)
                next_token_embed = self.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                break_flag = break_flag | (next_token[:,0] == stop_token_index)
                generated_mask.append(~break_flag)
                next_token_probs.append(logits.softmax(dim=-1).max(dim=-1)[0])
                if torch.all(break_flag):
                    break

            output_list = list(tokens.cpu().numpy())
            # output_text = [tokenizer.decode(_).rstrip(stop_token) for _ in output_list]
            output_text = [self.tokenizer.decode(_) for _ in output_list]
            output_text = [sent.split('.')[0] for sent in output_text]
            generated_list.append(output_text)
            next_token_probs_list.append(torch.stack(next_token_probs, dim=1))
            generated_mask_list.append(generated_mask)
        return generated_list[0], next_token_probs_list[0], torch.stack(generated_mask_list[0], 1)

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, gpt_model='gpt2', cache_dir=None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(gpt_model, cache_dir=cache_dir)

        self.tokenizer.pad_token = '!'
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        elif mapping_type == MappingType.Transformer:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
        else:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


if __name__ == '__main__':
    device='cuda'
    from transformers import AutoTokenizer

    prefix_length=10
    model = ClipCaptionModel(prefix_length=prefix_length, clip_length=1, prefix_size= 512,
                 num_layers= 8, mapping_type = MappingType.MLP).to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = '!'
    captions = [['A man rides a bicycle across the river.', 'A man jumping his bicycle onto a bench.', 'A man standing in a kitchen with granite countertops.']]
    # data = tokenizer(captions, padding='longest', return_tensors="pt")
    # tokens = data['input_ids'].to(device)
    # b,l = tokens.size()
    # mask = data['attention_mask'].to(device)
    # mask = torch.cat([torch.ones((b, prefix_length),device=device), mask], dim=1)
    prefix = torch.randn(3, 512).to(device)
    outputs = model(prefix, captions)
    logits = outputs.logits[:, prefix_length - 1: -1]
    loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
