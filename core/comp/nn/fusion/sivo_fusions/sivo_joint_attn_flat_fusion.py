from functools import reduce
from typing import Optional, Dict

import torch
from allennlp.modules import Seq2VecEncoder

from core.comp.nn.attention import SeqAttention
from core.comp.nn.fusion.sivo_fusions.sivo_fusion import SeqinVecoutFusion
from core.comp.nn.op_mask import OpMaskJointConcatSelfAttention, OpMaskJointConcatSequential
from core.comp.nn.seq_reduce import SeqReduce
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.simple_merge_utils import get_merge_method


@SeqinVecoutFusion.register('op_mask_joint_concat_attention')
class SiVoOpMaskJointConcatAttentionFusion(SeqinVecoutFusion):
    """
    This attention fusion module execute self-attention on the sequence containing
    both add and del elements.
    Add and del elements will be op-masked, selected and truncated first and then
    concatenated as a joint sequence as input to self=attention module, producing a
    joint query feature to attend raw add and del encoder outputs respectively.
    Finally features from attending of add and del encoder outputs will be merged as
    a single fusion out.
    """
    def __init__(
            self,
            encoder_feature_dim: int,
            transformer_layer: int,
            seq_reduce: SeqReduce,
            query_attention: SeqAttention,
            hunk_reduce: Optional[SeqReduce] = None,
            hunk_mask: bool = False,
            hunk_reduce_before_attention: bool = False,
            type_embedding_dim: Optional[int] = None,
            transformer_dim_feedforward: int = 2048,
            transformer_head: int = 8,
            transformer_dropout: float = 0.1,
            transformer_activation: str = 'relu',
            op_feature_dim: int = 0,        # This param should not be configured
            merge_method: str ='add',
            reduce_before_attention: bool = True,
            out_proj_and_norm: bool = False,
            out_proj_in_dim: Optional[int] = None,
            out_proj_out_dim: Optional[int] = None,
            unmasked_seq_max_len: int = 1 << 64,  # large default length means do nothing
            keep_unmasked_order: bool = False,
            **kwargs
    ):
        super().__init__(encoder_feature_dim,
                         op_feature_dim,
                         **kwargs)

        self.op_mask_self_attention = OpMaskJointConcatSelfAttention(encoder_feature_dim, transformer_layer,
                                                                     transformer_dim_feedforward,
                                                                     transformer_head, transformer_dropout,
                                                                     transformer_activation,
                                                                     unmasked_seq_max_len,
                                                                     keep_unmasked_order)

        self.seq_reduce = seq_reduce
        self.hunk_reduce = hunk_reduce
        self.hunk_mask = hunk_mask
        self.query_attention = query_attention
        self.reduce_before_attention = reduce_before_attention
        self.hunk_reduce_before_attention = hunk_reduce_before_attention

        if not reduce_before_attention:
            mylogger.warning('SiVoOpMaskJointConcatAttentionFusion',
                             'This attention fusion module do not recommend "reduce_before_attention"=False, ' +
                             'because self-attention mask will not be used in query-back process, ' +
                             'which may incur unexpected behaviors.')

        assert merge_method in ['add', 'mul', 'cat', 'sub']
        self.merge_method = get_merge_method(merge_method)

        self.out_proj_and_norm = out_proj_and_norm
        if out_proj_and_norm:
            self.out_proj_dim = out_proj_out_dim
            # (confidence_check suggests to disable bias before normalization)
            self.out_proj = torch.nn.Linear(out_proj_in_dim, out_proj_out_dim, bias=False)
            self.out_norm = torch.nn.LayerNorm(out_proj_out_dim)

        if type_embedding_dim is not None:
            self.type_embedding = torch.nn.Parameter(torch.randn((2, type_embedding_dim)))
            # self.type_embedding_proj = torch.nn.Linear(feature_dim + type_embedding, feature_dim)
        else:
            self.type_embedding = None
            

    def _maybe_append_type_embedding(self,
                                     features,
                                     type_embedding_idx: int,
                                     expand_wrapping_dim_num: int = 2):
        if self.type_embedding is None:
            return features
        else:
            # Maybe: [type_embed] -> [batch, seq, type_embed]
            wrapping_dims = [1]*expand_wrapping_dim_num
            type_embedding_exp = self.type_embedding[type_embedding_idx].view(*wrapping_dims,-1)\
                                                                        .repeat(*features.shape[:expand_wrapping_dim_num], 1)
            type_embed_out = torch.cat((features, type_embedding_exp), dim=-1)
            return type_embed_out


    def _adapt_wrapping_dim(self, feature, mask1, mask2):
        org_shape = feature.shape
        wrapping_dim = len(org_shape) - 3
        dummy_batch_size = reduce(lambda x,y: x*y, org_shape[:wrapping_dim+1], 1)
        feature = feature.view(dummy_batch_size, org_shape[-2], org_shape[-1])
        mask1 = mask1.view(dummy_batch_size, org_shape[-2])
        mask2 = mask2.view(dummy_batch_size, org_shape[-2])
        return feature, mask1, mask2, org_shape


    def _reduce_wrapping_dim(self, features, org_shape, pad_mask):
        wrapping_dim = len(org_shape) - 3
        feature_dim = features.size(-1)
        assert wrapping_dim == 0 or self.hunk_reduce is not None

        for dim in range(wrapping_dim):
            features = features.view(-1, org_shape[wrapping_dim-dim], feature_dim)
            # todo: maybe other hunk reduction methods, like attention ?
            # NOTE: Behavior of hunk reducing has been updated and different from ver.65 and ver.66
            hunk_mask = self._get_hunk_mask(pad_mask.view(-1, org_shape[wrapping_dim-dim], org_shape[wrapping_dim-dim+1]))
            features = self.hunk_reduce(features, mask=hunk_mask)

        return features

    def _get_hunk_mask(self, pad_mask) -> torch.BoolTensor:
        """
        This method assumes a non-padded hunk must has more than three non-padded elements.
        """
        hunk_unmask_sum = pad_mask.int().sum(-1)
        return hunk_unmask_sum > 1


    def _hunk_reduce_before_attn(self, query_features, add_pad_mask, del_pad_mask):
        """
        This method is customized for hunk reduction before attention, and it will not call
        "self._reduce_wrapping_dim" for convenience.
        Thus its logic, which only reduce hunk dim, is not synchronized with the general implementation.
        """
        add_hunk_mask = self._get_hunk_mask(add_pad_mask)
        del_hunk_mask = self._get_hunk_mask(del_pad_mask)
        query_hunk_mask = add_hunk_mask | del_hunk_mask

        bsz, hunk_num = query_hunk_mask[:2]
        query_features = query_features.view(bsz, hunk_num, -1)
        query_features = self.hunk_reduce(query_features, query_hunk_mask)

        return query_features


    def forward(self,
        add_code_input: Dict[str, torch.Tensor],
        del_code_input: Dict[str, torch.Tensor],
        op_input: Dict[str, torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
    )-> Dict[str, torch.Tensor]:
        add_code_features, add_pad_mask = self.extract_feature_and_mask(add_code_input)
        del_code_features, del_pad_mask = self.extract_feature_and_mask(del_code_input)
        
        add_code_features = self._maybe_append_type_embedding(add_code_features, 0)
        del_code_features = self._maybe_append_type_embedding(del_code_features, 0)
        add_code_features, add_pad_mask, add_op_mask, add_org_shape = self._adapt_wrapping_dim(add_code_features, add_pad_mask, add_op_mask)
        del_code_features, del_pad_mask, del_op_mask, del_org_shape = self._adapt_wrapping_dim(del_code_features, del_pad_mask, del_op_mask)
        # assert len(add_code_features.shape) == len(del_code_features.shape) == 3, 'Feature must be 3D of shape: [batch, seq, dim]'

        self_attn_out, self_attn_mask = self.op_mask_self_attention(add_code_features, del_code_features,
                                                                    add_op_mask, del_op_mask)
        if self.reduce_before_attention:
            self_attn_query = self.seq_reduce(self_attn_out, self_attn_mask)
            self_attn_query = self._hunk_reduce_before_attn(self_attn_query,
                                                            add_pad_mask,
                                                            del_pad_mask) if self.hunk_reduce_before_attention else self_attn_query
            add_fused_out = self.query_attention(self_attn_query, add_code_features, add_code_features, add_pad_mask)
            del_fused_out = self.query_attention(self_attn_query, del_code_features, del_code_features, del_pad_mask)
        else:
            # NOTE: hunk reduce before attention will not work when "reduce_before_attention=False".
            add_fused_out = self.query_attention(self_attn_out, add_code_features, add_code_features, add_pad_mask)
            del_fused_out = self.query_attention(self_attn_out, del_code_features, del_code_features, del_pad_mask)
            add_fused_out = self.seq_reduce(add_fused_out, self_attn_mask)
            del_fused_out = self.seq_reduce(del_fused_out, self_attn_mask)

        if not self.hunk_reduce_before_attention:
            add_fused_out = self._reduce_wrapping_dim(add_fused_out, add_org_shape, add_pad_mask)
            del_fused_out = self._reduce_wrapping_dim(del_fused_out, del_org_shape, del_pad_mask)

        fused_out = self.merge_method(add_fused_out, del_fused_out)

        if self.out_proj_and_norm:
            fused_out = self.out_norm(self.out_proj(fused_out))

        return {
            'encoder_outputs': fused_out,
            'source_mask': None
        }

    def get_output_dim(self) -> int:
        if self.out_proj_and_norm:
            return self.out_proj_dim
        else:
            return self.encoder_feature_dim     # Note this dim is not always correct