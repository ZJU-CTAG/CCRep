import torch

from allennlp.common.registrable import Registrable
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import LstmSeq2VecEncoder

from utils.calc import nanmean


class SeqReduce(Registrable, torch.nn.Module):
    def __init__(self):
        super(SeqReduce, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        raise NotImplementedError

    def mask(self, x: torch.Tensor, mask: torch.Tensor, fill_value):
        mask = mask.bool().unsqueeze(-1)
        return torch.masked_fill(x, ~mask, fill_value)


@SeqReduce.register('cls')
class ClsSeqReduce(SeqReduce):
    def __init__(self, cls_idx=0):
        super(ClsSeqReduce, self).__init__()
        self.cls_idx = cls_idx

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        assert len(x.shape) == 3, f'Input sequence must be 3D, but got shape: {x.shape}'

        return x[:, self.cls_idx, :]


@SeqReduce.register('max')
class MaxSeqReduce(SeqReduce):
    def __init__(self):
        super(MaxSeqReduce, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        assert len(x.shape) == 3, f'Input sequence must be 3D, but got shape: {x.shape}'

        if mask is not None:
            x = self.mask(x, mask, -float('inf'))

        return torch.max(x, dim=1).values


@SeqReduce.register('avg')
class AvgSeqReduce(SeqReduce):
    def __init__(self):
        super(AvgSeqReduce, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        assert len(x.shape) == 3, f'Input sequence must be 3D, but got shape: {x.shape}'

        if mask is not None:
            x = self.mask(x, mask, -float('nan'))
            return nanmean(x, dim=1)
        else:
            return torch.mean(x, dim=1)


@SeqReduce.register('lstm')
class LstmSeqReduceWrapper(SeqReduce):
    def __init__(self,
                 feature_dim: int,
                 lstm_hidden_size: int,
                 lstm_layer: int,
                 lstm_dropout: float = 0.3,
                 lstm_bidirectional: bool = True,
                 ):
        super().__init__()
        self.lstm = LstmSeq2VecEncoder(feature_dim,
                                       lstm_hidden_size,
                                       lstm_layer,
                                       dropout=lstm_dropout,
                                       bidirectional=lstm_bidirectional)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        return self.lstm(x, mask)