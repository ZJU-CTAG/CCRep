from typing import Dict, Optional

import torch
from allennlp.modules import Seq2VecEncoder
from allennlp.nn.util import add_positional_features

from core.comp.nn.attention import SeqAttention
from core.comp.nn.fusion.sivo_fusions.sivo_fusion import SeqinVecoutFusion
from core.comp.nn.seq_reduce import SeqReduce
from utils.allennlp_utils.simple_merge_utils import get_merge_method


@SeqinVecoutFusion.register('flat_line_align_joint_concat_attention')
class SiVoFlatLineAlignJointConcatAttentionFusion(SeqinVecoutFusion):
    def __init__(self,
                 encoder_feature_dim: int,
                 line_max_tokens: int,
                 max_lines: int,
                 line_token_seq_reduce: SeqReduce,
                 line_feature_seq_reduce: SeqReduce,
                 query_attention: Optional[SeqAttention] = None,
                 query_attention_merge_method: str = 'add',
                 transformer_layer: int = 2,
                 transformer_dim_feedforward: int = 2048,
                 transformer_head: int = 8,
                 transformer_dropout: float = 0.1,
                 transformer_activation: str = 'relu',
                 positional_encoding: str = 'sincos',
                 drop_tokenizer_head_tail_token: bool = True,
                 trainable_token_separator: bool = True,
                 insert_trainable_cls_token: bool = False,
                 out_proj_dim: Optional[int] = None,
                 op_feature_dim: int = 0,   # This param should not be configured
                 **kwargs):
        super().__init__(encoder_feature_dim, op_feature_dim, **kwargs)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=encoder_feature_dim,
                                                         nhead=transformer_head,
                                                         dim_feedforward=transformer_dim_feedforward,
                                                         dropout=transformer_dropout,
                                                         activation=transformer_activation)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer,
                                                       num_layers=transformer_layer,
                                                       norm=torch.nn.LayerNorm(encoder_feature_dim))

        self.line_token_seq_reduce = line_token_seq_reduce
        self.line_feature_seq_reduce = line_feature_seq_reduce
        self.line_max_tokens = line_max_tokens
        self.max_lines = max_lines
        self.positional_encoding = positional_encoding
        self.drop_tokenizer_head_tail_token = drop_tokenizer_head_tail_token

        self.add_del_separator_embedding = torch.nn.Parameter(torch.randn(encoder_feature_dim,)) \
                                           if trainable_token_separator else torch.zeros((encoder_feature_dim,))

        self.out_proj = torch.nn.Linear(encoder_feature_dim, out_proj_dim, bias=False) if out_proj_dim is not None else None
        self.cls_token_embedding = torch.nn.Parameter(torch.randn(encoder_feature_dim,)) if insert_trainable_cls_token else None

        self.query_attention = query_attention
        self.query_attention_merge = get_merge_method(query_attention_merge_method)

    def _truncate_feature_and_line_idx_seq(self, feature_seqs, line_idx_seqs):
        min_seq_len = min(feature_seqs.size(1), line_idx_seqs.size(1))
        return feature_seqs[:, :min_seq_len, :], line_idx_seqs[:, :min_seq_len]


    def maybe_add_pos_encoding(self, feature):
        if self.positional_encoding == 'sincos':
            return feature + add_positional_features(feature)
        elif self.positional_encoding == 'none':
            return feature
        else:
            raise ValueError(self.positional_encoding)


    def maybe_out_proj(self, out_features):
        if self.out_proj is not None:
            return self.out_proj(out_features)
        else:
            return out_features


    def pytorch_transformer_forward(self, features, mask):
        features = self.maybe_add_pos_encoding(features)
        features = features.transpose(0,1).contiguous()
        forward_features = self.transformer(features, src_key_padding_mask=~mask)
        return forward_features.transpose(0,1).contiguous()


    def rearrange_by_line_idx(self, code_features, line_idx):
        # Padded tensors to be filled
        # Allocate one more line to store scattered padding zeros
        bsz, feature_dim = code_features.size(0), code_features.size(-1)
        code_line_features = torch.zeros((bsz, (self.max_lines + 1) * self.line_max_tokens, feature_dim), device=code_features.device)

        # Because there are padded zeros in line indexes, we have to shift real line block by one
        # and make all padded zeros mapped to the first block, so that we can easily drop the first
        # block to get rid of padded 'zero' line indexes
        # Shape: [bsz, max_lines, line_max_tokens, feature_dim]
        line_idx_exp = line_idx.unsqueeze(-1).repeat((1, 1, feature_dim))
        code_line_features = torch.scatter(code_line_features, 1, line_idx_exp, code_features)\
                                      .view(bsz, self.max_lines + 1, self.line_max_tokens, feature_dim)[:,1:,:,:]

        return code_line_features


    def maybe_query_back(self, fused_feature,
                         add_code_feautres, add_pad_mask,
                         del_code_feautres, del_pad_mask):
        if self.query_attention is not None:
            add_attn_out = self.query_attention.forward(fused_feature, add_code_feautres, add_code_feautres, add_pad_mask)
            del_attn_out = self.query_attention.forward(fused_feature, del_code_feautres, del_code_feautres, del_pad_mask)
            return self.query_attention_merge(add_attn_out, del_attn_out)
        else:
            return fused_feature


    def forward(self,
        add_code_input: Dict[str, torch.Tensor],
        del_code_input: Dict[str, torch.Tensor],
        op_input: Dict[str, torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        add_code_features, add_pad_mask = self.extract_feature_and_mask(add_code_input)
        del_code_features, del_pad_mask = self.extract_feature_and_mask(del_code_input)

        # Remove '<s>' and '</s>' tokens from sequence
        # NOTE: Here exists a tiny bug where "-1" may not be </s> but <pad>,
        #       but there is no problem with it because </s> will never has a matched
        #       line idx. Only problem is </s> may leave in the sequence but <s> not.
        if self.drop_tokenizer_head_tail_token:
            add_code_features = add_code_features[:, 1:-1, :]
            add_pad_mask = add_pad_mask[:, 1:-1]
            del_code_features = del_code_features[:, 1:-1, :]
            del_pad_mask = del_pad_mask[:, 1:-1]

        # Ensure len(idx) <= len(src) when calling 'torch.scatter'
        add_code_features, add_line_idx = self._truncate_feature_and_line_idx_seq(add_code_features, add_line_idx)
        del_code_features, del_line_idx = self._truncate_feature_and_line_idx_seq(del_code_features, del_line_idx)

        # Padded tensors to be filled.
        # Allocate one more line to store scattered padding zeros.
        bsz, feature_dim = add_code_features.size(0), add_code_features.size(-1)
        add_code_line_features = self.rearrange_by_line_idx(add_code_features, add_line_idx)
        del_code_line_features = self.rearrange_by_line_idx(del_code_features, del_line_idx)

        # Use norm to compute mask, because unfilled positions are all zeros, thus norm=0
        add_code_token_mask = torch.norm(add_code_line_features, 1, dim=-1) != 0
        del_code_token_mask = torch.norm(del_code_line_features, 1, dim=-1) != 0
        sep_code_token_mask = torch.ones((bsz, self.max_lines, 1), dtype=torch.int, device=add_code_token_mask.device).bool()
        sep_code_line_features = self.add_del_separator_embedding.view(1, 1, 1, -1).repeat((bsz, self.max_lines, 1, 1))

        if self.cls_token_embedding is None:
            code_features_to_be_cat = (add_code_line_features, sep_code_line_features, del_code_line_features)
            code_mask_to_be_cat = (add_code_token_mask, sep_code_token_mask, del_code_token_mask)
        else:
            cls_code_line_features = self.cls_token_embedding.view(1, 1, 1, -1).repeat((bsz, self.max_lines, 1, 1))
            code_features_to_be_cat = (cls_code_line_features, add_code_line_features, sep_code_line_features, del_code_line_features)
            cls_code_token_mask = torch.ones((bsz, self.max_lines, 1), dtype=torch.int, device=add_code_token_mask.device).bool()
            code_mask_to_be_cat = (cls_code_token_mask, add_code_token_mask, sep_code_token_mask, del_code_token_mask)

        joint_code_line_mask = torch.cat(code_mask_to_be_cat, dim=-1).view(bsz*self.max_lines, -1)
        joint_code_features = torch.cat(code_features_to_be_cat, dim=2).view(bsz*self.max_lines, -1, feature_dim)

        # shape: [bsz, max_lines, feature_dim] -> [bsz, feature_dim]
        joint_line_features = self.pytorch_transformer_forward(joint_code_features, joint_code_line_mask)
        joint_line_features = self.line_token_seq_reduce(joint_line_features, joint_code_line_mask)

        # If either add or del has one scattered token, it is a valid line
        joint_line_mask = (add_code_token_mask.float().norm(1, -1) +
                           del_code_token_mask.float().norm(1, -1)) != 0
        joint_line_features = joint_line_features.view(bsz, self.max_lines, -1)
        joint_diff_features = self.line_feature_seq_reduce(joint_line_features, joint_line_mask)

        joint_diff_features = self.maybe_query_back(joint_diff_features,
                                                    add_code_features, add_pad_mask,
                                                    del_code_features, del_pad_mask)
        joint_diff_features = self.maybe_out_proj(joint_diff_features)
        return {
            'encoder_outputs': joint_diff_features,
            'source_mask': None
        }

