from typing import Dict, Optional

import torch

from core.comp.nn.fusion.diff_siso_fusions.diff_siso_fusion import DiffSeqinSeqoutFusion
from core.comp.nn.fusion.sivo_fusions.sivo_joint_attn_line_align_fusion import SiVoFlatLineAlignJointConcatAttentionFusion
from utils.allennlp_utils.simple_merge_utils import get_merge_method


@DiffSeqinSeqoutFusion.register('diff_flat_line_align')
class DiffSiSoFLAFusion(DiffSeqinSeqoutFusion):
    def __init__(self,
                 encoder_feature_dim: int,
                 sivo_FLA_fusion: SiVoFlatLineAlignJointConcatAttentionFusion,
                 fusion_as_decoder_input_method: str = 'as_init',
                 out_dim: Optional[int] = None,
                 op_feature_dim: int = 0):
        super().__init__(encoder_feature_dim, op_feature_dim)
        self.sivo_FLA_fusion = sivo_FLA_fusion
        self.fusion_as_decoder_input_method = fusion_as_decoder_input_method
        self.out_dim = out_dim

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
        sivo_fused_out = self.sivo_FLA_fusion.forward(add_code_input, del_code_input,
                                                      op_input,
                                                      add_op_mask, del_op_mask,
                                                      add_line_idx, del_line_idx)
        sivo_fused_features = sivo_fused_out['encoder_outputs']
        diff_code_features, diff_pad_mask = self.extract_feature_and_mask(diff_input)

        fusion_outs =  {
            'source_mask': diff_pad_mask,
        }

        # process fused out features
        if self.fusion_as_decoder_input_method == 'as_init':
            fusion_outs['decoder_init_state'] = sivo_fused_features
        elif self.fusion_as_decoder_input_method == 'context_cat':
            fusion_outs['fused_cat_context'] = sivo_fused_features
        elif self.fusion_as_decoder_input_method == 'merge':
            raise NotImplementedError
        else:
            # If not 'as_init' or 'merge', guess it is a algorithmic merge method
            fused_features_exp = sivo_fused_features.unsqueeze(1).repeat((1, diff_code_features.size(1), 1))
            diff_code_features = get_merge_method(self.fusion_as_decoder_input_method)(
                diff_code_features, fused_features_exp
            )

        fusion_outs['encoder_outputs'] = diff_code_features
        return fusion_outs

    def get_output_dim(self) -> int:
        if self.out_dim is not None:
            return self.out_dim
        else:
            # Guess a output dim, configure this if needed
            return self.encoder_feature_dim * 2