from typing import List, Union, Dict

from utils import GlobalLogger as mylogger

def replace_var_with_placeholder(code_input: Union[List,str],
                                 placeholder_mapping: Dict[str,str],
                                 joiner: str = '\n',
                                 keep_mapped_var: bool = True,
                                 minimum_replace_length: int = 2,
                                 verbose: bool = False):
    replaced_mapping = {}
    code_input_is_list = type(code_input) is list
    if code_input_is_list:
        code_input = joiner.join(code_input)

    for mapped, mapping in placeholder_mapping.items():
        if len(mapped) < minimum_replace_length:
            if verbose:
                mylogger.warning('replace_var_with_placeholder',
                                 f'Mapping identifier({mapped}) with len < minimum_len={minimum_replace_length}')
            continue
        if code_input.find(mapped) != -1:
            replaced_mapping[mapped] = mapping
        if keep_mapped_var:
            mapping = f'{mapped}{mapping}'
        code_input = code_input.replace(mapped, mapping)

    if code_input_is_list:
        code_input = code_input.split(joiner)
    return code_input, replaced_mapping

def revert_replaceholder_as_var(output: str,
                                placeholder_mapping: Dict[str,str],
                                mode: str = 'as_tokens',
                                joiner: str = ' '):
    assert mode in ['as_tokens', 'as_sentence']
    if mode == 'tokens':
        revert_mapping = {val:key for key,val in placeholder_mapping.items()}
        tokens = output.split(joiner)
        reverted_tokens = []
        for token in tokens:
            if token in revert_mapping:
                reverted_tokens.append(revert_mapping[token])
            else:
                reverted_tokens.append(token)
        return joiner.join(reverted_tokens)
    else:
        for map_tgt, map_src in placeholder_mapping.items():
            output = output.replace(map_src, map_tgt)
        return output