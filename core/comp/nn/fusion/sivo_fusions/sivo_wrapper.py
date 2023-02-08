from typing import Optional, Dict

import torch

from core.comp.nn.fusion.sivo_fusions.sivo_fusion import SeqinVecoutFusion
from core.comp.nn.fusion.sivo_fusions.sivo_joint_attn_flat_fusion import SiVoOpMaskJointConcatAttentionFusion
from core.comp.nn.fusion.sivo_fusions.sivo_joint_attn_line_align_fusion import SiVoFlatLineAlignJointConcatAttentionFusion


@SeqinVecoutFusion.register('op_mask_line_align_hybrid_add_wrapper')
class SiVoLineAlignOpMaskHybridAddWrapperFusion(SeqinVecoutFusion):
    def __init__(self,
                 encoder_feature_dim: int,
                 op_mask_joint_attention: SiVoOpMaskJointConcatAttentionFusion,
                 line_align_joint_attention: SiVoFlatLineAlignJointConcatAttentionFusion,
                 reproject_before_add: bool = False,
                 reproject_dropout: Optional[float] = None,
                 op_feature_dim: int = 0,   # This param should not be configured
                 **kwasrgs):
        super().__init__(encoder_feature_dim, op_feature_dim, **kwasrgs)

        self.op_mask_joint_attention = op_mask_joint_attention
        self.line_align_joint_attention = line_align_joint_attention
        self.reproject_before_add = reproject_before_add

        if reproject_before_add:
            self.op_mask_reproject = torch.nn.Linear(encoder_feature_dim, encoder_feature_dim, bias=False)
            self.op_mask_norm = torch.nn.LayerNorm(encoder_feature_dim)
            self.line_align_reproject = torch.nn.Linear(encoder_feature_dim, encoder_feature_dim, bias=False)
            self.line_align_norm = torch.nn.LayerNorm(encoder_feature_dim)
            self.reproject_dropout = reproject_dropout


    def forward(self,
        add_code_input: Dict[str, torch.Tensor],
        del_code_input: Dict[str, torch.Tensor],
        op_input: Dict[str, torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        op_mask_attn_out = self.op_mask_joint_attention(add_code_input, del_code_input,
                                                        op_input,
                                                        add_op_mask, del_op_mask,
                                                        None, None)
        line_align_attn_out = self.line_align_joint_attention(add_code_input, del_code_input,
                                                              op_input,
                                                              None, None,
                                                              add_line_idx, del_line_idx)

        op_mask_features = op_mask_attn_out['encoder_outputs']
        line_align_features = line_align_attn_out['encoder_outputs']

        if self.reproject_before_add:
            op_mask_features = self.op_mask_norm(self.op_mask_reproject(op_mask_features))
            line_align_features = self.line_align_norm(self.line_align_reproject(line_align_features))

            if self.reproject_dropout is not None:
                op_mask_features = torch.dropout(op_mask_features, self.reproject_dropout, self.training)
                line_align_features = torch.dropout(line_align_features, self.reproject_dropout, self.training)

        return {
            'encoder_outputs': op_mask_features + line_align_features,
            'source_mask': None
        }