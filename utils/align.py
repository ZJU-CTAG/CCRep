# encoding=utf-8

import difflib
from typing import List


def _heuristic_replace_match(a_tokens: List[str], b_tokens: List[str]):
    diff_seqs = []
    a_len = len(a_tokens)
    b_len = len(b_tokens)
    delta_len = max(a_len - b_len, b_len - a_len)
    if a_len != b_len:
        head_ratio = difflib.SequenceMatcher(None, a_tokens[0], b_tokens[0]).quick_ratio()
        tail_ratio = difflib.SequenceMatcher(None, a_tokens[-1], b_tokens[-1]).quick_ratio()
        if head_ratio >= tail_ratio:
            if a_len > b_len:
                b_tokens += [""] * delta_len
            else:
                a_tokens += [""] * delta_len
        else:
            if a_len > b_len:
                b_tokens = [""] * delta_len + b_tokens
            else:
                a_tokens = [""] * delta_len + a_tokens
    assert len(a_tokens) == len(b_tokens)
    for at, bt in zip(a_tokens, b_tokens):
        if at == "":
            diff_seqs.append([at, bt, "insert"])
        elif bt == "":
            diff_seqs.append([at, bt, "delete"])
        else:
            diff_seqs.append([at, bt, "replace"])
    return diff_seqs


def construct_diff_sequence(a: List[str], b: List[str]) -> List[List[str]]:
    diff_seqs = []
    diff = difflib.SequenceMatcher(None, a, b)

    for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
        a_tokens = a[a_i:a_j]
        b_tokens = b[b_i:b_j]
        if op == "delete":
            for at in a_tokens:
                diff_seqs.append([at, "", op])
        elif op == "insert":
            for bt in b_tokens:
                diff_seqs.append(["", bt, op])
        elif op == "equal":
            for at, bt in zip(a_tokens, b_tokens):
                diff_seqs.append([at, bt, op])
        else:
            # replace
            diff_seqs += _heuristic_replace_match(a_tokens, b_tokens)

    return diff_seqs


if __name__ == '__main__':
    a = 'a b c d a b c'.split(' ')
    b = 'a b c d e f g c'.split(' ')
    aligned = construct_diff_sequence(a, b)
    print(aligned)
