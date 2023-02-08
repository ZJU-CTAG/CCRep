from typing import Tuple, List
import torch
import difflib

from allennlp.data.fields import TextField, TensorField, ListField
from utils import GlobalLogger as mylogger

_no_change_count = 0

def diff_mask_by_strs(differ: difflib.Differ,
                      add_strs: List[str],
                      del_strs: List[str],
                      op_mask_attend_first_token: bool = True,
                      ) -> Tuple[TensorField, TensorField]:
    diff = differ.compare(del_strs, add_strs)

    changed_token_count = 0
    add_mask, del_mask = [], []
    for i, token in enumerate(diff):
        # check if always attend to first token, to avoid NaN caused by
        # full 'False' mask
        # NOTE: Set "op_mask_attend_first_token" to True may cause
        # len(op_mask) != len(tokens) when first token not match.
        if i == 0:
            # First lines equal, attend it on both sides
            if not token.startswith('+ ') and not token.startswith('- ') and op_mask_attend_first_token:
                add_mask.append(1)
                del_mask.append(1)
                continue
            # Else do regular rountine, because op-mask must be non-empty

        # added token, means this token has changed in add seqeuence
        if token.startswith('+ '):
            changed_token_count += 1
            add_mask.append(1)
        # removed token, means this token has changed in del seqeuence
        elif token.startswith('- '):
            changed_token_count += 1
            del_mask.append(1)
        # otherwise this token did not change
        # Fix: token not appears in either side
        elif not token.startswith('? '):
            add_mask.append(0)
            del_mask.append(0)

    if changed_token_count == 0:
        global _no_change_count
        _no_change_count += 1
        mylogger.warning('diff_mask_by_strs',
                         f'No changed token detected (NO. {_no_change_count}) with attend_first={op_mask_attend_first_token}!\nA: {del_strs}\nB:{add_strs}')

    return TensorField(torch.Tensor(add_mask)), \
           TensorField(torch.Tensor(del_mask))


def diff_mask_by_text_field(differ: difflib.Differ,
                            add_text_field: TextField,
                            del_text_field: TextField,
                            op_mask_attend_first_token: bool = True,
                            ) -> Tuple[TensorField, TensorField]:
    add_tokens = [token.text for token in add_text_field.tokens]
    del_tokens = [token.text for token in del_text_field.tokens]

    return diff_mask_by_strs(differ, add_tokens, del_tokens, op_mask_attend_first_token)


def diff_and_gen_op_mask(differ:difflib.Differ,
                         add_field, del_field,
                         op_mask_attend_first_token: bool = True):

    def _recur_gen_list_field_mask(_add_field, _del_field):
        assert len(_add_field) == len(_del_field), \
            f'len(add_field)={len(_add_field)} != len(del_field)={len(_del_field)}'
        _add_masks, _del_masks = [], []
        for _add_subfield, _del_subfield in zip(_add_field, _del_field):
            if type(_add_subfield) == ListField:
                _add_mask, _del_mask = _recur_gen_list_field_mask(_add_subfield, _del_subfield)
            elif type(_add_subfield) == TextField:
                _add_mask, _del_mask = diff_mask_by_text_field(differ,
                                                               _add_subfield, _del_subfield,
                                                               op_mask_attend_first_token) # self._diff_mask(_add_subfield, _del_subfield)
            else:
                raise TypeError(f'Unknown type of element in ListField: {type(_add_subfield)}')
            _add_masks.append(_add_mask)
            _del_masks.append(_del_mask)

        _add_mask_field = ListField(_add_masks)
        _del_mask_field = ListField(_del_masks)
        return _add_mask_field, _del_mask_field

    if type(add_field) == ListField:
        add_mask_field, del_mask_field = _recur_gen_list_field_mask(add_field, del_field)
    elif type(add_field) == TextField:
        add_mask_field, del_mask_field = diff_mask_by_text_field(differ, add_field, del_field,
                                                                 op_mask_attend_first_token)
    else:
        raise TypeError(f'Unknown type of element in ListField: {type(add_field)}')

    return add_mask_field, del_mask_field


def extract_lines_and_op_mask_from_marks(diff_tokens, diff_marks, max_len=-1):
    add_tokens, del_tokens = [], []
    add_mask, del_mask = [], []

    for token, mark in zip(diff_tokens, diff_marks):
        if mark == 2:
            add_tokens.append(token)
            del_tokens.append(token)
            add_mask.append(0)
            del_mask.append(0)
        elif mark == 1:
            del_tokens.append(token)
            del_mask.append(1)
        elif mark == 3:
            add_tokens.append(token)
            add_mask.append(1)

        if max_len != -1 and (len(add_tokens) == max_len or len(del_tokens) == max_len):
            break
    assert sum(add_mask) + sum(del_mask) != 0, f'No changed tokens when extract op-mask from marks, ' \
                                               f'which may lead to NaN problem. Diff tokens: {diff_tokens}'
    return add_tokens, del_tokens, add_mask, del_mask