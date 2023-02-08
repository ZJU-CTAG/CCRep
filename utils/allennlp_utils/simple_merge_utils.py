import torch


def get_merge_method(merge_method):
    if merge_method == 'add':
        return lambda a,b: a + b
    elif merge_method == 'sub':
        return lambda a,b: a - b
    elif merge_method == 'rsub':
        return lambda a,b: b - a
    elif merge_method == 'mul':
        return lambda a,b: a*b
    elif merge_method == 'cat':
        return lambda a,b: torch.cat((a,b), dim=-1)
    elif merge_method == 'avg':
        return lambda a,b: (a + b) / 2
    else:
        raise ValueError