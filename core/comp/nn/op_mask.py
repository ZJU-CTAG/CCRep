from typing import Tuple, Optional

import torch

from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder, GruSeq2SeqEncoder
from allennlp.nn.util import add_positional_features


class OpMaskJointSeq2Seq(torch.nn.Module):
    def __init__(self,
                 unmasked_seq_max_len: int,
                 keep_unmasked_order: bool = False,
                 type_embedding: Optional[int] = None,
                 feature_dim: Optional[int] = None):
        super(OpMaskJointSeq2Seq, self).__init__()

        self.unmasked_seq_max_len = unmasked_seq_max_len
        self.keep_unmasked_order = keep_unmasked_order


    def collect_unmasked_elems(self,
                               feature: torch.Tensor,
                               mask: torch.BoolTensor,
                               type_embedding_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keep_unmasked_order:
            bsz, seq_len = mask.shape
            shifts = reversed(torch.arange(1, seq_len+1, device=mask.device)).unsqueeze(0).repeat(bsz, 1)
            # set masked positions to 0
            # else reversed index to do stable sort on unmasked items
            shifted_mask = shifts * mask
            sorted_mask, sorted_mask_idxes = shifted_mask.sort(dim=1, descending=True)
            sorted_mask = sorted_mask.bool()
        else:
            # sequence dim sort
            sorted_mask, sorted_mask_idxes = mask.sort(dim=1, descending=True)

        sorted_mask_idxes = sorted_mask_idxes.unsqueeze(-1).repeat(1, 1, feature.size(-1))
        sorted_feature = torch.gather(feature, 1, sorted_mask_idxes)

        # truncate sequence with maximum-unmasked length
        # also consider max_length constrain of unmasked sequence
        batch_max_unmasked_len = min(max(sorted_mask.int().sum(1)).item(),
                                     self.unmasked_seq_max_len)
        sorted_mask = sorted_mask[:, :batch_max_unmasked_len]
        sorted_feature = sorted_feature[:, :batch_max_unmasked_len, :]

        return sorted_feature, sorted_mask


    def forward(self,
                add_code_features: torch.Tensor,
                del_code_features: torch.Tensor,
                add_op_mask: torch.Tensor,
                del_op_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class OpMaskJointConcatSelfAttention(OpMaskJointSeq2Seq):
    """
    Compare to common op_mask self-attention, this self-attention will simultaneously
    receive add and del code features as input, and generate joint representations of
    code change from add and del codes.
    This module will first use mask to obtain changed tokens in add and del codes, and
    then concatenate changed add and del tokens as input to self-attention module to
    produce queries .
    """
    def __init__(
            self,
            feature_dim: int,
            transformer_layer: int,
            transformer_dim_feedforward: int = 2048,
            transformer_head: int = 8,
            transformer_dropout: float = 0.1,
            transformer_activation: str = 'relu',
            unmasked_seq_max_len: int = 1 << 64,    # large default length means do nothing
            keep_unmasked_order: bool = False,
            pos_encoding: Optional[str] = None,
            **kwargs
    ):
        super().__init__(unmasked_seq_max_len, keep_unmasked_order, feature_dim)

        self.encoder_feature_dim = feature_dim
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim,
                                                         nhead=transformer_head,
                                                         dim_feedforward=transformer_dim_feedforward,
                                                         dropout=transformer_dropout,
                                                         activation=transformer_activation)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer,
                                                       num_layers=transformer_layer,
                                                       norm=torch.nn.LayerNorm(feature_dim))
        self.attn_head_nums = transformer_head
        self.pos_encoding = pos_encoding


    def _maybe_add_pos_encoding(self, features: torch.Tensor):
        if self.pos_encoding is None:
            return features
        elif self.pos_encoding == 'sincos':
            pos_features = add_positional_features(features)
            return features + pos_features
        else:
            raise ValueError(self.pos_encoding)


    def _convert_pytorch_mask(self, mask):
        if mask is None:
            return None
        # pytorch transformer receives bool-type opposite mask as we do
        return ~mask.bool()


    def _transformer_forward(self,
                             features: torch.Tensor,
                             attn_mask: Optional[torch.Tensor] = None,
                             pad_mask: Optional[torch.Tensor] = None):
        attn_mask, pad_mask = self._convert_pytorch_mask(attn_mask), \
                              self._convert_pytorch_mask(pad_mask)
        features = self._maybe_add_pos_encoding(features)
        features = features.transpose(0,1).contiguous()
        # because pad mask is a part of attention mask, thus we don't use it
        transformer_out = self.transformer.forward(features, attn_mask, pad_mask)
        return transformer_out.transpose(0,1).contiguous()


    def forward(self,
                add_code_features: torch.Tensor,
                del_code_features: torch.Tensor,
                add_op_mask: torch.Tensor,
                del_op_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        add_sorted_features, add_sorted_mask = self.collect_unmasked_elems(add_code_features, add_op_mask, 0)
        del_sorted_features, del_sorted_mask = self.collect_unmasked_elems(del_code_features, del_op_mask, 1)

        joint_features = torch.cat((add_sorted_features, del_sorted_features), dim=1)
        joint_mask = torch.cat((add_sorted_mask, del_sorted_mask), dim=1)

        joint_out = self._transformer_forward(joint_features, pad_mask=joint_mask)

        return joint_out, joint_mask


class OpMaskJointConcatSequential(OpMaskJointSeq2Seq):
    def __init__(self,
                 feature_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 seq2seq_type: str,
                 dropout: float = 0.5,
                 bidirectional: bool = True,
                 unmasked_seq_max_len: int = 1 << 64,    # large default length means do nothing
                 keep_unmasked_order: bool = False):
        super().__init__(unmasked_seq_max_len, keep_unmasked_order)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if seq2seq_type == 'lstm':
            self.seq2seq = LstmSeq2SeqEncoder(feature_dim, hidden_dim, num_layers,
                                              dropout=dropout, bidirectional=bidirectional)
        elif seq2seq_type == 'gru':
            self.seq2seq = GruSeq2SeqEncoder(feature_dim, hidden_dim, num_layers,
                                             dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError(f'Not supported seq2seq type: {seq2seq_type}')


    def forward(self,
                add_code_features: torch.Tensor,
                del_code_features: torch.Tensor,
                add_op_mask: torch.Tensor,
                del_op_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        add_sorted_features, add_sorted_mask = self.collect_unmasked_elems(add_code_features, add_op_mask, 0)
        del_sorted_features, del_sorted_mask = self.collect_unmasked_elems(del_code_features, del_op_mask, 1)

        joint_features = torch.cat((add_sorted_features, del_sorted_features), dim=1)
        joint_mask = torch.cat((add_sorted_mask, del_sorted_mask), dim=1)

        joint_out = self.seq2seq(joint_features, joint_mask)

        return joint_out, joint_mask