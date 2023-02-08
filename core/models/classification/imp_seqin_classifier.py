from typing import Dict, Optional
import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric
from overrides import overrides

from core.comp.nn.fusion.sivo_fusions.sivo_fusion import SeqinVecoutFusion
from core.comp.nn.classifier import Classifier
from core.comp.nn.loss_func import LossFunc
from utils.metric import update_metric


@Model.register('imp_seqin_classifier')
class ImpCcMsgSeqinClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        # [Note] This classifier assumes code feature has been collapsed into a single vector representation before fusion
        code_encoder: Seq2SeqEncoder,
        fusion: SeqinVecoutFusion,
        classifier: Classifier,
        loss_func: LossFunc,
        msg_embedder: Optional[TextFieldEmbedder] = None,
        msg_encoder: Optional[Seq2VecEncoder] = None,
        op_embedder: TextFieldEmbedder = None,
        metric: Metric = None,
        # initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._code_embedder = code_embedder
        self._code_encoder = code_encoder
        self._msg_embedder = msg_embedder
        self._msg_encoder = msg_encoder
        self._op_embedder = op_embedder
        self._fusion = fusion
        self._classifier = classifier
        self._loss_func = loss_func
        self._metric = metric
        self.debug_forward_count = 0

        # initializer(self)

    def forward_features(self,
                         diff_add: TextFieldTensors,
                         diff_del: TextFieldTensors,
                         diff_op: TextFieldTensors = None,
                         msg: TextFieldTensors = None,
                         additional_feature: Optional[torch.Tensor] = None,
                         add_op_mask: Optional[torch.Tensor] = None,
                         del_op_mask: Optional[torch.Tensor] = None,
                         add_line_idx: Optional[torch.Tensor] = None,
                         del_line_idx: Optional[torch.Tensor] = None,
                         **kwargs) -> torch.Tensor:
        """
        Forward input features and output extracted cc features.
        """
        add_code_feature = self._encode_text_field(diff_add, self._code_embedder, self._code_encoder)
        del_code_feature = self._encode_text_field(diff_del, self._code_embedder, self._code_encoder)
        if diff_op is not None:
            op_features = self._op_embedder(diff_op)
        else:
            op_features = None

        cc_feature = self._fusion(add_code_feature, del_code_feature, op_features,
                                  add_op_mask, del_op_mask,
                                  add_line_idx, del_line_idx)
        cc_repre = cc_feature['encoder_outputs']

        if msg is not None:
            msg_feature = self._encode_text_field(msg, self._msg_embedder, self._msg_encoder)
            msg_repre = msg_feature['encoder_outputs']
            cc_repre = torch.cat((cc_repre, msg_repre), dim=-1)

        if additional_feature is not None:
            cc_repre = torch.cat((cc_repre, additional_feature), dim=-1)

        return cc_repre

    @overrides
    def forward(
        self,  # type: ignore
        diff_add: TextFieldTensors,
        diff_del: TextFieldTensors,
        diff_op: TextFieldTensors = None,
        msg: TextFieldTensors = None,
        additional_feature: Optional[torch.Tensor] = None,
        add_op_mask: Optional[torch.Tensor] = None,
        del_op_mask: Optional[torch.Tensor] = None,
        add_line_idx: Optional[torch.Tensor] = None,
        del_line_idx: Optional[torch.Tensor] = None,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        self.debug_forward_count += 1

        cc_repre = self.forward_features(diff_add, diff_del, diff_op, msg, additional_feature,
                                         add_op_mask, del_op_mask,
                                         add_line_idx, del_line_idx)

        probs, pred_idxes = self._classifier(cc_repre)
        result = {'probs': probs, 'pred': pred_idxes}

        if label is not None:
            loss = self._loss_func(probs, label)
            result['loss'] = loss
            update_metric(self._metric, pred_idxes, probs, label)

        return result


    def _encode_text_field(self, tokens: Dict[str, Dict[str, torch.Tensor]],
                           embedder: TextFieldEmbedder,
                           encoder: Seq2SeqEncoder) -> Dict[str, torch.Tensor]:
        '''
        Same as _encode() method of ComposedSeq2Seq.
        '''
        # Note: It assumes there is only one tensor in the dict.
        dim_num = -1
        for k1, d in tokens.items():
            for k2, t in d.items():
                dim_num = len(t.size())
                break
        # adapt multi-dimensional input
        num_wrapping_dim = dim_num - 2

        # shape: (batch_size, max_input_sequence_length)
        seq_mask = util.get_text_field_mask(tokens, num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_features = embedder(tokens, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = encoder(embedded_features, seq_mask)
        return {
            "source_mask": seq_mask,
            "encoder_outputs": encoder_outputs
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self._metric is not None:
            metric = self._metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(metric) != dict:
                metric_name = self._metric.__class__.__name__
                metric = {metric_name: metric}
            metrics.update(metric)
        return metrics