from typing import Dict, Optional, Tuple

import torch
from allennlp.common import Registrable


class SeqinVecoutFusion(Registrable, torch.nn.Module):
    def __init__(
        self,
        encoder_feature_dim: int,
        op_feature_dim: int,
        **kwargs
    ):
        super().__init__()
        self.encoder_feature_dim = encoder_feature_dim
        self.op_feature_dim = op_feature_dim

    def forward(self,
        add_code_input: Dict[str, torch.Tensor],
        del_code_input: Dict[str, torch.Tensor],
        op_input: Dict[str, torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
    )-> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def extract_feature_and_mask(self, code_input: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        return code_input['encoder_outputs'], code_input['source_mask']