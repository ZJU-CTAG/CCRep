from typing import List, Optional, Dict

import torch
from allennlp.data import Token, Vocabulary


def convert_str_tokens_to_line(str_tokens: List[str],
                               exclude_tokens=['@start@', '@end@'],
                               replace_token_map: Dict[str, str] = {},
                               merge_subtoken_method: str = 'none',
                               end_token: Optional[str] = None) -> str:
    if end_token is not None:
        try:
            end_index = str_tokens.index(end_token)
        except ValueError:
            end_index = -1
        str_tokens = str_tokens[:end_index]

    str_tokens = [replace_token_map[t] if t in replace_token_map else t for t in str_tokens]
    if merge_subtoken_method == 'none':
        return ' '.join([t for t in str_tokens if t not in exclude_tokens])
    elif merge_subtoken_method == 'codebert':
        res = ''
        for wordpiece in str_tokens:
            if wordpiece in exclude_tokens:
                continue
            if wordpiece[0] == 'Ġ':
                res += f' {wordpiece[1:]}'
            else:
                res += wordpiece
        return res


def convert_tokens_to_strs(tokens: List[Token],
                           exclude_tokens=['@start@', '@end@'],
                           replace_token_map: Dict[str, str] = {},
                           merge_subtoken_method: str = 'none',
                           end_token: Optional[str] = None) -> str:
    str_tokens = [t.text for t in tokens]
    return convert_str_tokens_to_line(str_tokens, exclude_tokens,
                                      replace_token_map,
                                      merge_subtoken_method,
                                      end_token)


def convert_prediction_tokens_to_ids(token_list: List[str],
                                     vocab: Vocabulary,
                                     namespace: str,
                                     excluded_tokens: List[str] = [],
                                     return_tensor=True):
    token_ids = []
    for token in token_list:
        if token not in excluded_tokens:
            token_ids.append(vocab.get_token_index(token, namespace))

    if return_tensor:
        return torch.LongTensor(token_ids)
    else:
        return token_ids


def convert_prediction_ids_to_tokens(id_list: List[int],
                                     _vocab: Vocabulary,
                                     namespace: str = 'code_tokens',
                                     excluded_tokens: List[str] = []):
    excluded_token_ids = [_vocab.get_token_index(exc_token, namespace) for exc_token in excluded_tokens]
    token_ids = []
    for token_id in id_list:
        if token_id in excluded_token_ids:
            continue
        token_ids.append(_vocab.get_token_from_index(token_id, namespace))

    return token_ids