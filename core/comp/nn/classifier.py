from typing import List, Tuple

import torch
from torch import nn

from allennlp.common import Registrable

from core.comp.nn.mlp import mlp_block
from utils import GlobalLogger as mylogger

class Classifier(Registrable, nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self._num_class = num_class

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @return: logits and predicted label indexes
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        return self._num_class

    def get_exp_input_dim(self) -> int:
        raise NotImplementedError


@Classifier.register('linear_softmax')
class LinearSoftmaxClassifier(Classifier):
    def __init__(self,
                 in_feature_dim: int,
                 out_feature_dim: int,
                 hidden_dims: List[int],
                 activations: List[str],
                 dropouts: List[float],
                 ahead_feature_dropout: float = 0.,
                 log_softmax: bool = False):   # actaul layer_num = len(hidden_dims) + 1
        super().__init__(out_feature_dim)

        assert len(hidden_dims) == len(activations) == len(dropouts)

        in_dims = [in_feature_dim, *hidden_dims]
        out_dims = [*hidden_dims, out_feature_dim]
        activations = [*activations, None]      # no activation at last last output layer
        dropouts = [*dropouts, None]            # no dropout at last output layer

        layers = [
            mlp_block(in_dim, out_dim, activation, dropout)
            for in_dim, out_dim, activation, dropout in
            zip(in_dims, out_dims, activations, dropouts)
        ]
        self._layers = nn.Sequential(*layers)
        self._in_feature_dim = in_feature_dim
        self._ahead_feature_dropout = torch.nn.Dropout(ahead_feature_dropout)
        self.softmax = torch.nn.LogSoftmax(dim=-1) if log_softmax else torch.nn.Softmax(dim=-1)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._ahead_feature_dropout(feature)
        logits = self._layers(feature)
        probs = self.softmax(logits)
        pred_idxes = torch.max(probs, dim=-1).indices
        # mylogger.debug('LinearSoftmaxClassifier.forward',
        #                f'probs:{probs}')
        return probs, pred_idxes

    def get_exp_input_dim(self) -> int:
        return self._in_feature_dim


@Classifier.register('linear_sigmoid')
class LinearSigmoidClassifier(LinearSoftmaxClassifier):
    def __init__(self,
                 in_feature_dim: int,
                 hidden_dims: List[int],
                 activations: List[str],
                 dropouts: List[float],
                 ahead_feature_dropout: float = 0.,
                 out_dim: int = 1):   # actaul layer_num = len(hidden_dims) + 1
        super().__init__(in_feature_dim,
                         out_dim,
                         hidden_dims,
                         activations,
                         dropouts,
                         ahead_feature_dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._ahead_feature_dropout(feature)
        logits = self._layers(feature)
        logits = self.sigmoid(logits).squeeze(-1)
        pred_idxes = (logits > 0.5).long()
        # mylogger.debug('linear_sigmoid',
        #                f'logits: {logits}\npred_idxes:{pred_idxes}')
        return logits, pred_idxes
