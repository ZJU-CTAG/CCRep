from typing import List, Dict

from allennlp.data import Vocabulary
from overrides import overrides
import torch
import numpy

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric

from core.comp.metric.bin.B_Norm import b_norm, b_norm_new
from utils.allennlp_utils.cmg_id_token_utils import convert_str_tokens_to_line


@Metric.register('bleu_norm')
class BNorm(Metric):
    """
    This version of B-Norm implementation only uses script to calculate B-Norm
    and simply report the result of B-Norm script.
    """
    def __init__(
        self,
        vocab: Vocabulary,
        token_namespace: str,
        excluded_tokens: List[str] = ['@start@', '@end@'],
        replace_token_map: Dict[str, str] = {},
        cal_bleu_per_batch: bool = False,
        subtoken_merge_method: str = 'none',
    ) -> None:
        self.vocab = vocab
        self.token_namespace = token_namespace
        self.excluded_tokens = excluded_tokens
        self.replace_token_map = replace_token_map

        self.preds = []
        self.refs = []
        self.bleu_vals = []
        self.cal_bleu_per_batch = cal_bleu_per_batch
        self.subtoken_merge_method = subtoken_merge_method

    @overrides
    def reset(self) -> None:
        self.refs.clear()
        self.preds.clear()
        self.bleu_vals.clear()

    def _convert_indice_to_tokens(self, indices_tensor):
        token_list = []
        for i in range(len(indices_tensor)):
            tokens = []
            for idx in indices_tensor[i]:
                token = self.vocab.get_token_from_index(idx.item(), self.token_namespace)
                if token not in self.excluded_tokens:
                    if token in self.replace_token_map:
                        tokens.append(self.replace_token_map[token])
                    else:
                        tokens.append(token)

            token_str = convert_str_tokens_to_line(tokens, self.excluded_tokens, self.replace_token_map, self.subtoken_merge_method)
            token_list.append(token_str)
        return token_list

    @overrides
    def __call__(
        self,  # type: ignore
        predictions: torch.LongTensor,
        gold_targets: torch.LongTensor,
    ) -> None:
        """
        Update predictions and references of B-Norm.
        """
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)
        pred_tokens = self._convert_indice_to_tokens(predictions)
        gold_tokens = self._convert_indice_to_tokens(gold_targets)
        if self.cal_bleu_per_batch:
            bleu = b_norm_new(gold_tokens, pred_tokens)
            self.bleu_vals.append(bleu)
        else:
            self.preds.extend(pred_tokens)
            self.refs.extend(gold_tokens)
        # bleu = b_norm(gold_tokens, pred_tokens)
        # self.bleu_vals.append(bleu)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self.cal_bleu_per_batch:
            b_norm_val = numpy.mean(self.bleu_vals)
        else:
            b_norm_val = b_norm_new(self.refs, self.preds)
        if reset:
            self.reset()
        return {"BLEU": b_norm_val}




