from typing import Optional, List

import numpy
import torch
from allennlp.data import Tokenizer, Token

from utils.align import construct_diff_sequence


def pretrained_transformer_tokenize_and_filter_line(line,
                                                    tokenizer: Tokenizer,
                                                    keep_tokenizer_token = False):
    tokenized_line = tokenizer.tokenize(line)
    # drop '<s>' and '</s>'
    if not keep_tokenizer_token:
        return tokenized_line[1:-1]
    else:
        return tokenized_line


def pretrained_transformer_postprocess_code(tokenized_code_seq,
                                            line_idx,
                                            tokenizer_token_kept = False):
    """
    Add '<s>' and '</s>' token at beginning and end of code sequence.
    """
    # NOTE: <s> and </s> must be sliced out because they are not considered when indexing line!
    if not tokenizer_token_kept:
        tokenized_code_seq.insert(0, Token('<s>'))
        tokenized_code_seq.append(Token('</s>'))
        # # Add zero line-idx for tokenizer special token when not keeping them
        # # NOTE: This modification od
        # line_idx.insert(0, 0)
        # line_idx.append(0)

    # todo: When the whole diff has len=0, insert a dummy empty token and set its line_idx=0, to make it filtered as padding during line aligning
    if len(tokenized_code_seq) == 0:
        tokenized_code_seq.insert(1, Token('Ġ'))
        line_idx.insert(1, 0)

    return tokenized_code_seq, line_idx


def generate_line_gather_idx(tokens, line_idx, line_max_tokens):
    token_shift = numpy.arange(0, len(tokens))
    # Here add an additional "1" to distinguish between padded zero
    # NOTE: Line indexes must minus 'line_max_tokens' to get real line indexes
    token_line_gather_idxes = token_shift + line_idx * line_max_tokens + line_max_tokens
    return token_line_gather_idxes.tolist()


def generate_filled_line_idx(tokens, filled_val):
    filled_line_idx = [filled_val] * len(tokens)
    return filled_line_idx


def return_line_aligned_diff(add_aligned_lines, del_aligned_lines, add_line_idxes, del_line_idxes,
                             diff_max_tokens,
                             tokenizer_token_kept = False):
    assert len(add_aligned_lines) == len(add_line_idxes) <= diff_max_tokens + 3
    assert len(del_aligned_lines) == len(del_line_idxes) <= diff_max_tokens + 3
    add_aligned_lines, add_line_idxes = pretrained_transformer_postprocess_code(add_aligned_lines, add_line_idxes, tokenizer_token_kept)
    del_aligned_lines, del_line_idxes = pretrained_transformer_postprocess_code(del_aligned_lines, del_line_idxes, tokenizer_token_kept)
    # add_line_idxes = add_line_idxes[:diff_max_tokens]
    # del_line_idxes = del_line_idxes[:diff_max_tokens]

    return {
        'add': add_aligned_lines,
        'del': del_aligned_lines,
        'add_line_idx': torch.LongTensor(add_line_idxes),
        'del_line_idx': torch.LongTensor(del_line_idxes),
    }


def return_line_aligned_single_stream(aligned_lines, line_idxes,
                                      diff_max_tokens,
                                      tokenizer_token_kept = False):
    assert len(aligned_lines) == len(line_idxes) <= diff_max_tokens + 3
    aligned_lines, line_idxes = pretrained_transformer_postprocess_code(aligned_lines, line_idxes, tokenizer_token_kept)

    return aligned_lines, line_idxes


def maybe_add_line_separator(tokens, line_separator):
    if line_separator is not None:
        tokens.append(Token(line_separator))
    return tokens


def get_empty_line_placeholders(tokenizer: Tokenizer,
                                keep_tokenizer_token: bool,
                                line_separator: Optional[str],
                                empty_line_placeholder: str) -> List[Token]:
    placeholders = pretrained_transformer_tokenize_and_filter_line(
        line = empty_line_placeholder,
        tokenizer=tokenizer,
        keep_tokenizer_token=keep_tokenizer_token
    )
    if line_separator is not None:
        placeholders.append(Token(line_separator))

    return placeholders


def generate_aligned_lines_and_idxes(diff,
                                     data_type,
                                     tokenizer: Tokenizer,
                                     line_separator: Optional[str] = None,
                                     empty_line_placeholder: Optional[str] = None,
                                     line_max_tokens: int = 64,
                                     diff_max_tokens: int = 256,
                                     max_lines: int = 64,
                                     insert_empty_line_placeholder: bool = False,
                                     keep_tokenizer_head_tail_token: bool = False,
                                     align_equal_lines: bool = True,
                                     set_first_line_equal: bool = False):
    assert data_type in ['del', 'add']
    aligned_tokenized_lines, aligned_line_idxes = [], []
    line_idx = -1

    for hunk in diff:
        lines = construct_diff_sequence(hunk['removed_code'], hunk['added_code'])

        for i, line in enumerate(lines):
            line_idx += 1
            if line_idx == max_lines or len(aligned_tokenized_lines) >= diff_max_tokens:
                return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)

            if i == 1 and set_first_line_equal:
                line[-1] = 'equal'

            line_empty_flag = False
            # No remove, only add
            if line[-1] == 'insert':
                if data_type == 'add':
                    tokenized_line = pretrained_transformer_tokenize_and_filter_line(line[1], tokenizer, keep_tokenizer_head_tail_token)
                    tokenized_line = maybe_add_line_separator(tokenized_line, line_separator)
                    # pre-check length constraint before extending
                    if len(aligned_tokenized_lines) + len(tokenized_line) >= diff_max_tokens:
                        return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)
                    else:
                        aligned_tokenized_lines.extend(tokenized_line)
                        aligned_line_idxes.extend(generate_line_gather_idx(tokenized_line, line_idx, line_max_tokens))
                else:
                    line_empty_flag = True

            # No add, only remove
            elif line[-1] == 'delete':
                if data_type == 'del':
                    tokenized_line = pretrained_transformer_tokenize_and_filter_line(line[0], tokenizer, keep_tokenizer_head_tail_token)
                    tokenized_line = maybe_add_line_separator(tokenized_line, line_separator)
                    # pre-check length constraint before extending
                    if len(aligned_tokenized_lines) + len(tokenized_line) >= diff_max_tokens:
                        return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)
                    else:
                        aligned_tokenized_lines.extend(tokenized_line)
                        aligned_line_idxes.extend(generate_line_gather_idx(tokenized_line, line_idx, line_max_tokens))
                else:
                    line_empty_flag = True

            # Lines replace or match line whatever 'equal' or 'replace'
            elif line[-1] == 'replace' or align_equal_lines:
                fetch_idx = 0 if data_type == 'del' else 1
                tokenized_line = pretrained_transformer_tokenize_and_filter_line(line[fetch_idx], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_line = maybe_add_line_separator(tokenized_line, line_separator)
                # pre-check length constraint before extending
                if len(aligned_tokenized_lines) + len(tokenized_line) >= diff_max_tokens:
                    return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)
                else:
                    aligned_tokenized_lines.extend(tokenized_line)
                    aligned_line_idxes.extend(generate_line_gather_idx(tokenized_line, line_idx, line_max_tokens))

            # Line equals and do not align them
            else:
                tokenized_line = pretrained_transformer_tokenize_and_filter_line(line[0], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_line = maybe_add_line_separator(tokenized_line, line_separator)
                if len(aligned_tokenized_lines) + len(tokenized_line) >= diff_max_tokens:
                    return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)
                else:
                    aligned_tokenized_lines.extend(tokenized_line)
                    # Use line_idx=0 as idx placeholder to filter these equal lines
                    aligned_line_idxes.extend(generate_filled_line_idx(tokenized_line, 0))


            # Maybe insert placeholders for empty add/del line
            if insert_empty_line_placeholder:
                assert empty_line_placeholder is not None
                line_placeholders = get_empty_line_placeholders(tokenizer, keep_tokenizer_head_tail_token,
                                                                line_separator, empty_line_placeholder) # [Token(empty_line_placeholder), Token(line_separator)]
                if line_empty_flag:
                    # Here extend line_empty placeholders may break line constraint, thus need some tolerance
                    aligned_tokenized_lines.extend(line_placeholders)
                    aligned_line_idxes.extend(generate_line_gather_idx(line_placeholders, line_idx, line_max_tokens))

    return return_line_aligned_single_stream(aligned_tokenized_lines, aligned_line_idxes, diff_max_tokens, keep_tokenizer_head_tail_token)


def align_lines_in_diff_two_streams(diff,
                                    tokenizer: Tokenizer,
                                    line_separator: Optional[str] = None,
                                    empty_line_placeholder: Optional[str] = None,
                                    line_max_tokens: int = 64,
                                    diff_max_tokens: int = 256,
                                    max_lines: int = 64,
                                    insert_empty_line_placeholder: bool = False,
                                    keep_tokenizer_head_tail_token: bool = False,
                                    align_equal_lines: bool = True,
                                    set_first_line_equal: bool = False):
    # Independently process add and del aligned lines stream
    add_aligned_lines, add_line_idxes = generate_aligned_lines_and_idxes(diff, 'add',
                                                                         tokenizer,
                                                                         line_separator, empty_line_placeholder,
                                                                         line_max_tokens, diff_max_tokens, max_lines,
                                                                         insert_empty_line_placeholder, keep_tokenizer_head_tail_token,
                                                                         align_equal_lines,
                                                                         set_first_line_equal)
    del_aligned_lines, del_line_idxes = generate_aligned_lines_and_idxes(diff, 'del',
                                                                         tokenizer,
                                                                         line_separator, empty_line_placeholder,
                                                                         line_max_tokens, diff_max_tokens, max_lines,
                                                                         insert_empty_line_placeholder, keep_tokenizer_head_tail_token,
                                                                         align_equal_lines,
                                                                         set_first_line_equal)
    return {
        'add': add_aligned_lines,
        'del': del_aligned_lines,
        'add_line_idx': torch.LongTensor(add_line_idxes),
        'del_line_idx': torch.LongTensor(del_line_idxes),
    }


def align_lines_in_diff_jointly(diff,
                                tokenizer: Tokenizer,
                                line_separator: Optional[str] = None,
                                empty_line_placeholder: Optional[str] = None,
                                line_max_tokens: int = 64,
                                diff_max_tokens: int = 256,
                                max_lines: int = 64,
                                insert_empty_line_placeholder: bool = False,
                                keep_tokenizer_head_tail_token: bool = False,
                                align_equal_lines: bool = True,
                                set_first_line_equal: bool = False):
    aligned_add_lines, aligned_del_lines = [], []
    add_line_idxes, del_line_idxes = [], []

    line_idx = -1
    # Here hunks are flattened
    for hunk in diff:
        aligned_lines = construct_diff_sequence(hunk['removed_code'], hunk['added_code'])

        for i, aligned_line in enumerate(aligned_lines):
            line_idx += 1
            if line_idx == max_lines or len(aligned_add_lines) >= diff_max_tokens or len(aligned_del_lines) >= diff_max_tokens:
                return return_line_aligned_diff(aligned_add_lines, aligned_del_lines, add_line_idxes, del_line_idxes,
                                                diff_max_tokens, keep_tokenizer_head_tail_token)

            add_line_empty_flag, del_line_empty_flag = False, False
            # No remove, only add
            if aligned_line[-1] == 'insert':
                tokenized_add_line = pretrained_transformer_tokenize_and_filter_line(aligned_line[1], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_add_line = maybe_add_line_separator(tokenized_add_line, line_separator)
                if len(aligned_add_lines) + len(tokenized_add_line) >= diff_max_tokens:
                    return return_line_aligned_diff(aligned_add_lines, aligned_del_lines, add_line_idxes, del_line_idxes,
                                                    diff_max_tokens, keep_tokenizer_head_tail_token)

                aligned_add_lines.extend(tokenized_add_line)
                add_line_idxes.extend(generate_line_gather_idx(tokenized_add_line, line_idx, line_max_tokens))
                del_line_empty_flag = True

            # No add, only remove
            elif aligned_line[-1] == 'delete':
                tokenized_del_line = pretrained_transformer_tokenize_and_filter_line(aligned_line[0], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_del_line = maybe_add_line_separator(tokenized_del_line, line_separator)
                if len(aligned_del_lines) + len(tokenized_del_line) >= diff_max_tokens:
                    return return_line_aligned_diff(aligned_add_lines, aligned_del_lines, add_line_idxes, del_line_idxes,
                                                    diff_max_tokens, keep_tokenizer_head_tail_token)

                aligned_del_lines.extend(tokenized_del_line)
                del_line_idxes.extend(generate_line_gather_idx(tokenized_del_line, line_idx, line_max_tokens))
                add_line_empty_flag = True

            # Line match, whatever 'equal' or 'replace'
            else:
                # add
                tokenized_add_line = pretrained_transformer_tokenize_and_filter_line(aligned_line[1], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_add_line = maybe_add_line_separator(tokenized_add_line, line_separator)
                # remove
                tokenized_del_line = pretrained_transformer_tokenize_and_filter_line(aligned_line[0], tokenizer, keep_tokenizer_head_tail_token)
                tokenized_del_line = maybe_add_line_separator(tokenized_del_line, line_separator)

                if len(aligned_add_lines) + len(tokenized_add_line) >= diff_max_tokens or \
                   len(aligned_del_lines) + len(tokenized_del_line) >= diff_max_tokens:
                    return return_line_aligned_diff(aligned_add_lines, aligned_del_lines, add_line_idxes, del_line_idxes,
                                                    diff_max_tokens, keep_tokenizer_head_tail_token)

                aligned_add_lines.extend(tokenized_add_line)
                add_line_idxes.extend(generate_line_gather_idx(tokenized_add_line, line_idx, line_max_tokens))
                aligned_del_lines.extend(tokenized_del_line)
                del_line_idxes.extend(generate_line_gather_idx(tokenized_del_line, line_idx, line_max_tokens))

            # todo: Handle case where "align_equal_lines" = False...

            # Maybe insert placeholders for empty add/del line
            if insert_empty_line_placeholder:
                assert empty_line_placeholder is not None
                line_placeholders = get_empty_line_placeholders(tokenizer, keep_tokenizer_head_tail_token,
                                                                line_separator, empty_line_placeholder) # [Token(empty_line_placeholder), Token(line_separator)]
                if add_line_empty_flag:
                    aligned_add_lines.extend(line_placeholders)
                    add_line_idxes.extend(generate_line_gather_idx(line_placeholders, line_idx, line_max_tokens))
                if del_line_empty_flag:
                    aligned_del_lines.extend(line_placeholders)
                    del_line_idxes.extend(generate_line_gather_idx(line_placeholders, line_idx, line_max_tokens))

    return return_line_aligned_diff(aligned_add_lines, aligned_del_lines, add_line_idxes, del_line_idxes,
                                    diff_max_tokens, keep_tokenizer_head_tail_token)


if __name__ == "__main__":
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer
    test_diff = [{'added_code': ['',
   '@ Override',
   'public String toString ( ) {',
   'return "ChangeMessage { "',
   '+ "key = " + key',
   '+ " , author = " + author',
   '+ " , writtenOn = " + writtenOn',
   '+ " , patchset = " + patchset',
   '+ " , message = [ " + message',
   '+ " ] } " ;',
   '}'],
  'removed_code': []},
 {'added_code': ['import com . google . common . base . Joiner ;',
   'private final ImmutableList < ChangeMessage > changeMessages ;',
   'this . changeMessages =',
   'ChangeNotes . MESSAGE BY TIME . immutableSortedCopy ( changeMessages ) ;',
   'for ( ChangeMessage m : this . changeMessages ) {',
   'checkArgument ( m . getKey ( ) . getParentKey ( ) . equals ( change . getId ( ) ) ) ;',
   'Map < ChangeMessage . Key , ChangeMessage > as =',
   'changeMessageMap ( bundleA . changeMessages ) ;',
   'Map < ChangeMessage . Key , ChangeMessage > bs =',
   'changeMessageMap ( bundleB . changeMessages ) ;',
   'return ;'],
  'removed_code': ['import com . google . auto . value . AutoValue ;',
   'import com . google . common . collect . Multiset ;',
   'import com . google . common . collect . Ordering ;',
   'import com . google . common . collect . TreeMultiset ;',
   'import com . google . gerrit . common . Nullable ;',
   'import com . google . gerrit . common . TimeUtil ;',
   'import com . google . gerrit . reviewdb . client . Account ;',
   'import com . google . gwtorm . client . IntKey ;',
   'private final ImmutableMap < ChangeMessage . Key , ChangeMessage > changeMessages ;',
   'this . changeMessages = ImmutableMap . copyOf ( changeMessageMap ( changeMessages ) ) ;',
   'for ( ChangeMessage . Key k : this . changeMessages . keySet ( ) ) {']},
 {'added_code': ['import static java . util . concurrent . TimeUnit . SECONDS ;',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   '<com>',
   'static final long MAX WINDOW MS = SECONDS . toMillis ( 3 ) ;'],
  'removed_code': ['import java . util . concurrent . TimeUnit ;',
   'private static final long TS WINDOW MS =',
   'TimeUnit . MILLISECONDS . convert ( 1 , TimeUnit . SECONDS ) ;',
   'checkState ( when . getTime ( ) - update . getWhen ( ) . getTime ( ) < = TS WINDOW MS ,',
   'private static final long MAX DELTA MS = 1000 ;',
   'private static final long MAX WINDOW MS = 5000 ;',
   '']},
 {'added_code': ['long maxMs = ChangeRebuilder . MAX WINDOW MS ;',
   'assertThat ( maxMs ) . isGreaterThan ( 1000L ) ;',
   'TestTimeUtil . resetWithClockStep ( maxMs * 2 , MILLISECONDS ) ;',
   'private void superWindowResolution ( ) {',
   'TestTimeUtil . setClockStep (',
   'ChangeRebuilder . MAX WINDOW MS * 2 , MILLISECONDS ) ;',
   'TimeUtil . nowTs ( ) ;',
   '}',
   '',
   'private void subWindowResolution ( ) {',
   'TestTimeUtil . setClockStep ( 1 , SECONDS ) ;'],
  'removed_code': ['TestTimeUtil . resetWithClockStep ( 1 , SECONDS ) ;',
   'private void subSecondResolution ( ) {',
   'TestTimeUtil . setClockStep ( 100 , MILLISECONDS ) ;',
   '+ " { 2009 - 09 - 30 17 : 00 : 00 . 0 } ! = { 2009 - 09 - 30 17 : 00 : 01 . 0 } " ,',
   '+ " { 2009 - 09 - 30 17 : 00 : 00 . 0 } ! = { 2009 - 09 - 30 17 : 00 : 01 . 0 } " ) ;',
   'public void diffChangesMixedSourcesRoundsTimestamp ( ) throws Exception {',
   'subSecondResolution ( ) ;',
   '+ " { 2009 - 09 - 30 17 : 00 : 00 . 0 } ! = { 2009 - 09 - 30 17 : 00 : 01 . 1 } " ,',
   '+ " { 2009 - 09 - 30 17 : 00 : 00 . 0 } ! = { 2009 - 09 - 30 17 : 00 : 01 . 2 } " ) ;',
   '<com>',
   'assertDiffs ( b1 , b2 ,']}]

    print('Preparing tokenizer...')
    tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base',
                                               max_length=32,
                                               tokenizer_kwargs={
                                                   'additional_special_tokens': ['<com>']
                                               })
    print('Aligning...')
    aligned_diff = align_lines_in_diff_jointly(test_diff, tokenizer,
                                               line_separator='<nl>',
                                               empty_line_placeholder='<empty>',
                                               line_max_tokens=32,
                                               diff_max_tokens=256,
                                               insert_empty_line_placeholder=True)
