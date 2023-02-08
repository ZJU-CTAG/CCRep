from typing import Dict, Optional, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, util
from allennlp_models.generation import SeqDecoder
from overrides import overrides

from core.comp.nn.fusion.diff_siso_fusions.diff_siso_fusion import DiffSeqinSeqoutFusion


@Model.register('imp_cc_hybrid_seq2seq')
class HybridCcModularSeq2Seq(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        msg_decoder: SeqDecoder,
        fusion: Optional[DiffSeqinSeqoutFusion] = None,
        op_embedder: Optional[TextFieldEmbedder] = None,
        # initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._code_embedder = code_embedder
        self._code_encoder = code_encoder
        self._msg_decoder = msg_decoder
        self._op_embedder = op_embedder
        self._fusion = fusion

    @overrides
    def forward(
            self,  # type: ignore
            diff: TextFieldTensors,
            diff_add: TextFieldTensors = None,
            diff_del: TextFieldTensors = None,
            diff_op: TextFieldTensors = None,
            msg: TextFieldTensors = None,
            add_op_mask: Optional[torch.Tensor] = None,
            del_op_mask: Optional[torch.Tensor] = None,
            add_line_idx: Optional[torch.Tensor] = None,
            del_line_idx: Optional[torch.Tensor] = None,
            meta_data: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        diff_state = self._encode(diff)
        if self._fusion is not None:
            add_state = self._encode(diff_add)
            del_state = self._encode(diff_del)
            if diff_op is not None:
                assert self.op_embedder is not None
                op_state = self.op_embedder(diff_op)
            else:
                op_state = None

            state = self._fusion(diff_state, add_state, del_state, op_state, add_op_mask, del_op_mask, add_line_idx, del_line_idx)
        else:
            state = diff_state

        output_dict = self._msg_decoder(state, msg, diff)
        if meta_data is not None:
            output_dict.update({
                'meta_data': meta_data
            })
        return output_dict

    def _encode(self, code_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Same as _encode() method of ComposedSeq2Seq.
        '''
        # shape: (batch_size, max_input_sequence_length)
        code_seq_mask = util.get_text_field_mask(code_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_code_input = self._code_embedder(code_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._code_encoder(embedded_code_input, code_seq_mask)
        return {
            "source_mask": code_seq_mask,
            "encoder_outputs": encoder_outputs
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._msg_decoder.get_metrics(reset)
        return metrics
