import torch
from torch import nn
import torch.nn.functional as F

class FiraCopyNet(nn.Module):

    def __init__(self,
                 source_embedding_size: int,
                 target_embedding_size: int,
                 hidden_size: int = 1024,
                 ):
        super(FiraCopyNet, self).__init__()
        self.source_embedding_size = source_embedding_size
        self.target_embedding_size = target_embedding_size
        self.hidden_size = hidden_size
        self.LinearSource = nn.Linear(self.source_embedding_size, self.hidden_size, bias=False)
        self.LinearTarget = nn.Linear(self.target_embedding_size, self.hidden_size, bias=False)
        self.LinearRes = nn.Linear(self.hidden_size, 1)
        self.LinearGate= nn.Linear(self.target_embedding_size, 2)

    def forward(self, source, target, source_mask):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(target)
        # size: [batch, target_len, source_len]
        copy_probs = self.LinearRes(F.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        copy_probs = torch.masked_fill(copy_probs, ~source_mask.unsqueeze(1), float('-inf'))
        gate = F.softmax(self.LinearGate(target), dim=-1)
        return copy_probs, gate