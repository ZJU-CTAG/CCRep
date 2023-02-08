############################################################################
# This module has the same function as DiffSiSoOpMaskFusion, but reuses the
# SiVoOpMaskFusion as a component.
############################################################################

from typing import Optional, Dict

import torch

from core.comp.nn.fusion.diff_siso_fusions.diff_siso_fusion import DiffSeqinSeqoutFusion
from core.comp.nn.fusion.sivo_fusions.sivo_joint_attn_flat_fusion import SiVoOpMaskJointConcatAttentionFusion
from core.comp.nn.mlp import mlp_block
from utils.allennlp_utils.simple_merge_utils import get_merge_method


@DiffSeqinSeqoutFusion.register('diff_op_mask_fix')
class DiffSiSoOpMaskJointConcatAttentionFixFusion(DiffSeqinSeqoutFusion):
    """
    Almost the same as "SiSoOpMaskJointConcatAttentionFusion", except for having one
    more argument "diff_input" in forward().
    Fused features, which is a vector, can be either initial states of decoder,
    or directly append to every step of the joint sequence (cat).
    """
    def __init__(self,
                 encoder_feature_dim: int,
                 sivo_op_mask_fusion: SiVoOpMaskJointConcatAttentionFusion,
                 fusion_as_decoder_input_method: str = 'as_init',
                 fusion_raw_merge_kwargs: Optional[Dict] = None,
                 fusion_out_norm_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 op_feature_dim: int = 0,  # This param should not be configured
                 **kwargs
                 ):
        super().__init__(
            encoder_feature_dim, op_feature_dim, **kwargs
        )

        self.sivo_op_mask_fusion = sivo_op_mask_fusion
        self.output_dim = output_dim

        # Adapt legacy 'append' method to 'cat'
        if fusion_as_decoder_input_method == 'append':
            fusion_as_decoder_input_method = 'cat'

        self.fusion_as_decoder_input_method = fusion_as_decoder_input_method
        if fusion_as_decoder_input_method == 'merge':
            self.fusion_raw_merge_layers = torch.nn.Sequential(*[
                mlp_block(fusion_raw_merge_kwargs['in_dims'][i],
                          fusion_raw_merge_kwargs['out_dims'][i],
                          fusion_raw_merge_kwargs['activations'][i],
                          fusion_raw_merge_kwargs['dropouts'][i])
                for i in range(len(fusion_raw_merge_kwargs['in_dims']))
            ])

        if fusion_as_decoder_input_method == 'mask_add':
            self.op_mask_out_proj = torch.nn.Linear(encoder_feature_dim, encoder_feature_dim, bias=False)
            self.op_mask_out_norm = torch.nn.LayerNorm(encoder_feature_dim)

        if fusion_out_norm_dim is not None:
            self.fusion_out_norm = torch.nn.LayerNorm(encoder_feature_dim)
        else:
            self.fusion_out_norm = None


    def forward(self,
        diff_input: Dict[str, torch.Tensor],
        add_code_input: Dict[str, torch.Tensor],
        del_code_input: Dict[str, torch.Tensor],
        op_input: Dict[str, torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused_features = self.sivo_op_mask_fusion.forward(
            add_code_input, del_code_input, op_input,
            add_op_mask, del_op_mask
        )['encoder_outputs']
        diff_code_features, diff_pad_mask = self.extract_feature_and_mask(diff_input)

        fusion_outs =  {
            'source_mask': diff_pad_mask
        }
        if self.fusion_as_decoder_input_method == 'as_init':
            fusion_outs['decoder_init_state'] = fused_features
        elif self.fusion_as_decoder_input_method == 'context_cat':
            fusion_outs['fused_cat_context'] = fused_features
        elif self.fusion_as_decoder_input_method == 'merge':
            fused_features_exp = fused_features.unsqueeze(1).repeat((1, diff_code_features.size(1), 1))
            diff_code_features = torch.cat((diff_code_features, fused_features_exp), dim=-1)
            diff_code_features = self.fusion_raw_merge_layers(diff_code_features)
        else:
            # If not 'as_init' or 'merge', guess it is a algorithmic merge method
            fused_features_exp = fused_features.unsqueeze(1).repeat((1, diff_code_features.size(1), 1))
            diff_code_features = get_merge_method(self.fusion_as_decoder_input_method)(
                diff_code_features, fused_features_exp
            )

        if self.fusion_out_norm is not None:
            diff_code_features = self.fusion_out_norm(diff_code_features)

        fusion_outs['encoder_outputs'] = diff_code_features
        return fusion_outs

    def get_output_dim(self) -> int:
        if self.output_dim is not None:       # manually set output dim
            return self.output_dim
        else:
            return self.encoder_feature_dim     # Note this dim is not always correct