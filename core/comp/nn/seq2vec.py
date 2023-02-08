from typing import Optional, List, Tuple, Union
import torch
import torch.nn.functional as F

from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer
from allennlp.modules.seq2vec_encoders import CnnEncoder, BertPooler
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import LstmSeq2VecEncoder

from utils.profiling import log_run_time_ms

@Seq2VecEncoder.register('transformer')
class TransformerSeq2VecEncoder(Seq2VecEncoder):
    """
    code_encoder:{
        type: "transformer",
        input_dim: code_embed_dim,
        num_layers: 2,
        feedforward_hidden_dim: 512,
        feedforward_hidden_dim: 6,
        positional_encoding: "sinusoidal",
        dropout_prob: 0.5,
        activation: "relu"
    },
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 feedforward_hidden_dim: int = 2048,
                 num_attention_heads: int = 8,
                 positional_encoding: Optional[str] = None,
                 positional_embedding_size: int = 512,
                 dropout_prob: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        self._seq2seq = PytorchTransformer(input_dim,
                                           num_layers,
                                           feedforward_hidden_dim,
                                           num_attention_heads,
                                           positional_encoding,
                                           positional_embedding_size,
                                           dropout_prob,
                                           activation)
        self._seq2vec = ClsPooler(embedding_dim=self._seq2seq.get_output_dim())

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        transformer_forward = self._seq2seq(tokens, mask)
        cls_feature = self._seq2vec(transformer_forward, mask)
        return cls_feature

    def get_input_dim(self) -> int:
        return self._seq2seq.get_input_dim()

    def get_output_dim(self) -> int:
        return self._seq2vec.get_output_dim()
