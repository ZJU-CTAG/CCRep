import difflib
from typing import Iterable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, Token

from utils.data_utils.line_align_utils import align_lines_in_diff_jointly, align_lines_in_diff_two_streams
from utils.data_utils.op_mask_utils import diff_and_gen_op_mask
from utils.file import read_dumped


def cal_LA_metric(diff: List[Dict]) -> int:
    LA = 0
    for hunk in diff:
        LA += len(hunk['added_code'])
    return LA


def cal_LD_metric(diff: List[Dict]) -> int:
    LD = 0
    for hunk in diff:
        LD += len(hunk['removed_code'])
    return LD


@DatasetReader.register("jit_dp_imp_base")
class ImpBaseJitDPDatasetReader(DatasetReader):
    '''
    Implicit code change dataset reader for jit-dp task.
    An instance must contain 'add', 'del' and 'label' fields.
    'Op' field is optional for aligned sequences, and it can
    be None for unaligned sequences.
    '''

    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            max_tokens: int = 128,  # only valid for 'flat' diff structure
            start_token: Optional[str] = None,
            end_token: Optional[str] = None,
            include_LA_metric: bool = False,
            include_LD_metric: bool = False,
            use_op_mask: bool = False,
            op_mask_attend_first_token: bool = True,
            lower_msg: bool = True,
            empty_code_token: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {"code_tokens": code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.msg_tokenizer = WhitespaceTokenizer()
        self.msg_token_indexers = {"msg_tokens": SingleIdTokenIndexer(namespace='msg_tokens', lowercase_tokens=lower_msg)}   # enable lower by default
        self.max_tokens = max_tokens

        self.start_token = start_token
        self.end_token = end_token
        self.include_LA_metric = include_LA_metric
        self.include_LD_metric = include_LD_metric

        self.use_op_mask = use_op_mask
        self.op_mask_attend_first_token = op_mask_attend_first_token
        self.empty_code_token = empty_code_token
        self.differ = difflib.Differ()


    def get_token_list_for_empty_code(self):
        """
        Process empty code input, to avoid NaN out of Transformer's forward.
        """
        if self.empty_code_token is None:
            tokens = self.code_tokenizer.tokenize('')
            # If len=2, it is probably PretrainedTransformerTokenizer that outputs <s> and </s>.
            # We add at least one non-functional token to tokens to prevent unmatched shape
            # between token_ids and mask caused by transformer indexer
            if len(tokens) == 2:
                tokens.insert(1, Token(''))
            # ensure not empty returned token list
            elif len(tokens) == 0:
                tokens.append(Token(''))
        else:
            # There is a empty code token, try to tokenize it with tokenizer, to
            # add some special tokens at head and tail
            tokens = self.code_tokenizer.tokenize(self.empty_code_token)

        return tokens


    def extract_add_del_code_hierarchy(self, diff) -> Tuple[List, List]:
        add_diff, del_diff = [], []
        for hunk in diff:
            add_lines_in_hunk, del_lines_in_hunk = [], []
            # list of string lines
            add_lines = hunk.get('added_code')
            del_lines = hunk.get('removed_code')

            for add_line in add_lines:
                # Filter emtpy line
                if add_line == '':
                    continue
                add_lines_in_hunk.append(add_line)
            for del_line in del_lines:
                # Filter emtpy line
                if del_line == '':
                    continue
                del_lines_in_hunk.append(del_line)

            # reserve code structure as nested lists
            add_diff.append(add_lines_in_hunk)
            del_diff.append(del_lines_in_hunk)

        return add_diff, del_diff


    def text_to_instance(self, commit_id, label, commit_msg, diff) -> Instance:
        raise NotImplementedError


    def _read(self, file_path: str) -> Iterable[Instance]:
        data_sections = read_dumped(file_path)

        for data_block in tqdm(zip(*data_sections), total=len(data_sections[0])):
            yield self.text_to_instance(*data_block)


@DatasetReader.register("jit_dp_imp_flat")
class ImpFlatJitDPDatasetReader(ImpBaseJitDPDatasetReader):
    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            max_tokens: int = None,
            start_token: Optional[str] = None,
            end_token: Optional[str] = None,
            include_LA_metric: bool = False,
            include_LD_metric: bool = False,
            use_op_mask: bool = False,
            op_mask_attend_first_token: bool = True,
            lower_code: bool = False,
            lower_msg: bool = False,
            hunk_separator: Optional[str] = None,
            line_joiner: str = ' ',
            empty_code_token: Optional[str] = None,
            **kwargs
    ):
        super().__init__(code_indexer, code_tokenizer, max_tokens,
                         start_token, end_token,
                         include_LA_metric, include_LD_metric,
                         use_op_mask, op_mask_attend_first_token,
                         lower_msg, empty_code_token,
                         **kwargs)
        self.lower_code = lower_code
        self.lower_msg = lower_msg
        self.hunk_separator = hunk_separator
        self.line_joiner = line_joiner
        self.tokenized_diff = []


    def _flatten_and_make_text_field(self, code) -> TextField:
        lines = []
        for hunk in code:
            for line in hunk:
                lines.append(line)
            if self.hunk_separator is not None:
                lines.append(self.hunk_separator)

        if len(lines) == 0:
            tokens = self.get_token_list_for_empty_code()
        else:
            lines_str = self.line_joiner.join(lines)
            tokens = self.code_tokenizer.tokenize(lines_str)

        return TextField(tokens[:self.max_tokens], self.code_token_indexers)


    def text_to_instance(self, commit_id, label, commit_msg, diff) -> Instance:
        # calculate metric before reformat diff
        LA = cal_LA_metric(diff) if self.include_LA_metric else None
        LD = cal_LD_metric(diff) if self.include_LD_metric else None


        add_diff, del_diff = self.extract_add_del_code_hierarchy(diff)
        diff_add_field = self._flatten_and_make_text_field(add_diff)
        diff_del_field = self._flatten_and_make_text_field(del_diff)

        tokenized_msg = self.msg_tokenizer.tokenize(commit_msg)
        msg_field = TextField(tokenized_msg, self.msg_token_indexers)

        fields = {
            'diff_add': diff_add_field,
            'diff_del': diff_del_field,
            'msg': msg_field
        }

        # make label as a field only when it is available
        if label is not None:
            # Make label a numerical field
            fields['label'] = LabelField(int(label), skip_indexing=True)
        # make LA and LD metric as a field only when it is available
        if LA is not None:
            fields['LA_metric'] = TensorField(torch.FloatTensor([LA]))
        if LD is not None:
            fields['LD_metric'] = TensorField(torch.FloatTensor([LD]))

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(self.differ,
                                                                        diff_add_field, diff_del_field,
                                                                        self.op_mask_attend_first_token)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)


@DatasetReader.register("jit_dp_imp_flat_line_align")
class ImpFlatLineAlignJitDPDatasetReader(ImpBaseJitDPDatasetReader):
    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            line_max_tokens: int = 64,
            diff_max_tokens: int = 256,
            max_lines: int = 64,
            line_separator: Optional[str] = None,
            empty_line_placeholder: Optional[str] = None,
            insert_empty_line_placeholder: bool = False,
            use_op_mask: bool = False,
            op_mask_attend_first_token: bool = True,
            include_LA_metric: bool = False,
            include_LD_metric: bool = False,
            lower_code: bool = False,
            lower_msg: bool = False,
            keep_tokenizer_head_tail_token: bool = False,
            jointly_align_add_del_lines: bool = True,
            **kwargs
    ):
        super().__init__(code_indexer, code_tokenizer, diff_max_tokens,
                         None, None,
                         include_LA_metric, include_LD_metric,
                         use_op_mask, op_mask_attend_first_token,
                         lower_msg,
                         **kwargs)
        self.lower_code = lower_code
        self.lower_msg = lower_msg

        self.line_separator = line_separator
        self.line_max_tokens = line_max_tokens
        self.diff_max_tokens = diff_max_tokens
        self.max_lines = max_lines
        self.empty_line_placeholder = empty_line_placeholder
        self.insert_empty_line_placeholder = insert_empty_line_placeholder
        self.keep_tokenizer_head_tail_token = keep_tokenizer_head_tail_token
        self.line_align_func = align_lines_in_diff_jointly if jointly_align_add_del_lines else \
                               align_lines_in_diff_two_streams

        self.tokenized_diff = []


    def text_to_instance(self, commit_id, label, commit_msg, diff) -> Instance:
        # calculate metric before reformat diff
        LA = cal_LA_metric(diff) if self.include_LA_metric else None
        LD = cal_LD_metric(diff) if self.include_LD_metric else None

        aligned_diff = self.line_align_func(diff, self.code_tokenizer,
                                            self.line_separator,
                                            self.empty_line_placeholder,
                                            self.line_max_tokens,
                                            self.diff_max_tokens,
                                            self.max_lines,
                                            self.insert_empty_line_placeholder,
                                            self.keep_tokenizer_head_tail_token)
        diff_add_field = TextField(aligned_diff['add'], self.code_token_indexers)
        diff_del_field = TextField(aligned_diff['del'], self.code_token_indexers)
        add_line_idx_field = TensorField(aligned_diff['add_line_idx'])
        del_line_idx_field = TensorField(aligned_diff['del_line_idx'])

        tokenized_msg = self.msg_tokenizer.tokenize(commit_msg)
        msg_field = TextField(tokenized_msg, self.msg_token_indexers)

        fields = {
            'diff_add': diff_add_field,
            'diff_del': diff_del_field,
            'msg': msg_field,
            'add_line_idx': add_line_idx_field,
            'del_line_idx': del_line_idx_field,
        }

        # make label as a field only when it is available
        if label is not None:
            # Make label a numerical field
            fields['label'] = LabelField(int(label), skip_indexing=True)
        # make LA and LD metric as a field only when it is available
        if LA is not None:
            fields['LA_metric'] = TensorField(torch.FloatTensor([LA]))
        if LD is not None:
            fields['LD_metric'] = TensorField(torch.FloatTensor([LD]))

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(self.differ,
                                                                        diff_add_field, diff_del_field,
                                                                        self.op_mask_attend_first_token)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)