import torch
from torch.nn import functional

from allennlp.common import Registrable

from utils import GlobalLogger as mylogger

class LossFunc(Registrable, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def align_pred_label_batch_size(self, pred, label):
        '''
        Call this method to align the batch dimension of predictions and
        labels, dealing with unmatched batch size in tail batch caused by
        different padding implementations of different fields (such as
        'TensorField' type labels will not be padded).
        # [Note]: This method assumes predictions and labels are matched
        #         at corresponding dimension, which may not be true.
        '''
        pred_size, label_size = pred.size(0), label.size(0)
        if pred_size == label_size:
            return pred, label
        else:
            smaller_batch_size = min(pred_size, label_size)
            return pred[:smaller_batch_size], \
                   label[:smaller_batch_size]


@LossFunc.register('binary_cross_entropy')
class BinaryCrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return functional.binary_cross_entropy(pred, label)


@LossFunc.register('cross_entropy')
class CrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return functional.cross_entropy(pred, label)


@LossFunc.register('nll')
class NllLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return functional.nll_loss(pred, label)


@LossFunc.register('mean_square')
class MeanSquareLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze()
        assert pred.size() == label.size(), f'MSE assumes logit and label have the same size,' \
                                             f'but got {pred.size()} and {label.size()}'

        return (pred - label) ** 2


@LossFunc.register('bce_logits')
class BCEWithLogitsLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        pred, label = self.align_pred_label_batch_size(pred, label)
        return self._loss(pred, label)


@LossFunc.register('bce')
class BCELoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCELoss()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        pred, label = self.align_pred_label_batch_size(pred, label)
        pred = pred.view(pred.size(0),)
        label = label.view(pred.size(0),)
        return self._loss(pred, label.float())  # float type tensor is expected for 'label'

