from typing import Optional, Dict

import torch

from core.comp.nn.fusion.diff_siso_fusions.diff_siso_fusion import DiffSeqinSeqoutFusion
from core.comp.nn.fusion.sivo_fusions.sivo_wrapper import SiVoOpMaskJointConcatAttentionFusion, \
    SiVoFlatLineAlignJointConcatAttentionFusion
from utils.allennlp_utils.simple_merge_utils import get_merge_method


@DiffSeqinSeqoutFusion.register('diff_op_mask_line_align_hybrid')
class DiffSiSoOpMaskFLAHybridFusion(DiffSeqinSeqoutFusion):
    def __init__(self,
                 encoder_feature_dim: int,
                 sivo_op_mask_fusion: SiVoOpMaskJointConcatAttentionFusion,
                 sivo_line_align_fusion: SiVoFlatLineAlignJointConcatAttentionFusion,
                 fusion_as_decoder_input_method: str = 'as_init',
                 reproject_before_merge: bool = False,
                 reproject_dropout: Optional[float] = None,
                 hybrid_merge_method: str = 'add',
                 hybrid_merge_dropout: float = 0.,
                 output_dim: Optional[int] = None,
                 op_feature_dim: int = 0):
        super().__init__(encoder_feature_dim, op_feature_dim)
        self.output_dim = output_dim
        self.sivo_line_align_fusion = sivo_line_align_fusion
        self.sivo_op_mask_fusion = sivo_op_mask_fusion
        self.fusion_as_decoder_input_method = fusion_as_decoder_input_method
        self.reproject_before_merge = reproject_before_merge

        self.hybrid_merge_dropout =torch.nn.Dropout(hybrid_merge_dropout)
        if hybrid_merge_method not in ['reduce_dim']:
            self.hybrid_merge_func = get_merge_method(hybrid_merge_method)
        elif hybrid_merge_method == 'reduce_dim':
            self.hybrid_merge_layer = torch.nn.Linear(encoder_feature_dim*2, encoder_feature_dim, bias=False)
            self.hybrid_merge_func = lambda a,b: self.hybrid_merge_layer(torch.cat((a,b), dim=-1))
        else:
            raise NotImplementedError(f"hybrid_merge_method={hybrid_merge_method}")

        if reproject_before_merge:
            self.op_mask_reproject = torch.nn.Linear(encoder_feature_dim, encoder_feature_dim, bias=False)
            self.op_mask_norm = torch.nn.LayerNorm(encoder_feature_dim)
            self.line_align_reproject = torch.nn.Linear(encoder_feature_dim, encoder_feature_dim, bias=False)
            self.line_align_norm = torch.nn.LayerNorm(encoder_feature_dim)
            self.reproject_dropout = reproject_dropout


    def merge_two_fused_out(self, op_mask_fused_out, line_align_fused_out):
        op_mask_features = op_mask_fused_out['encoder_outputs']
        line_align_features = line_align_fused_out['encoder_outputs']

        if self.reproject_before_merge:
            op_mask_features = self.op_mask_norm(self.op_mask_reproject(op_mask_features))
            line_align_features = self.line_align_norm(self.line_align_reproject(line_align_features))
            if self.reproject_dropout is not None:
                op_mask_features = torch.dropout(op_mask_features, self.reproject_dropout, self.training)
                line_align_features = torch.dropout(line_align_features, self.reproject_dropout, self.training)

        hybrid_merged = self.hybrid_merge_func(op_mask_features, line_align_features)
        return self.hybrid_merge_dropout(hybrid_merged)

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
        sivo_op_mask_fused_out = self.sivo_op_mask_fusion(
            add_code_input, del_code_input, op_input,
            add_op_mask, del_op_mask,
            add_line_idx, del_line_idx
        )
        sivo_line_align_fused_out = self.sivo_line_align_fusion(
            add_code_input, del_code_input, op_input,
            add_op_mask, del_op_mask,
            add_line_idx, del_line_idx
        )

        sivo_fused_features = self.merge_two_fused_out(sivo_op_mask_fused_out, sivo_line_align_fused_out)
        diff_features, diff_pad_mask = self.extract_feature_and_mask(diff_input)
        fusion_outs =  {
            'source_mask': diff_pad_mask
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
            fused_features_exp = sivo_fused_features.unsqueeze(1).repeat((1, diff_features.size(1), 1))
            diff_features = get_merge_method(self.fusion_as_decoder_input_method)(
                diff_features, fused_features_exp
            )

        fusion_outs['encoder_outputs'] = diff_features
        return fusion_outs

    def get_output_dim(self) -> int:
        if self.output_dim is not None:
            return self.output_dim
        else:
            # guess a probable output dimension
            return self.encoder_feature_dim * 2