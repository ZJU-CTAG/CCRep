####################################################################
#
#   V2 reader: Only keep code change diff.
#
####################################################################

import difflib
from typing import Iterable, Dict, List, Optional

import torch
from tqdm import tqdm

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, SequenceLabelField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, PretrainedTransformerTokenizer, Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from utils import GlobalLogger as mylogger
from utils.align import construct_diff_sequence
from utils.allennlp_utils.extract_utils import get_line_extractor
from utils.data_utils.cmg_data_utils import make_target_sequence_field
from utils.data_utils.line_align_utils import align_lines_in_diff_jointly, align_lines_in_diff_two_streams
from utils.file import load_json, read_dumped
from utils.data_utils.op_mask_utils import diff_and_gen_op_mask


def load_data_lines(path):
    """load lines from a file"""
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

# check if a diff start with code change
def is_edit_diff(diff):
    return diff.startswith('mmm')

def is_new_or_del_file_mode_edif_diff(diff):
    if not diff.startswith('deleted file') and not diff.startswith('new file'):
        return False
    code_lines = diff.split('<nl>')
    first_line = code_lines[2].strip()
    return first_line.startswith('mmm')


def _make_token_list(str_list: Optional[List[str]],
                     max_length: int = -1) -> Optional[List[Token]]:
    ret = [Token(token) for i, token in enumerate(str_list) if max_length == -1 or i < max_length]
    return ret

def _split_and_truncate(seq: Optional[str], max_length: int) -> Optional[str]:
    if seq is None:
        return None
    if max_length == -1:
        return seq
    else:
        return ' '.join(seq.split(' ')[:max_length])


def check_skip_diff_only_complete_code_change(diff):
    """Check if this diff should be skipped"""
    if not is_edit_diff(diff):
        return True
    # Filter diffs has non-code changes
    spec_tokens = ['binary files', 'similarity index', 'old mode', 'new mode', 'deleted file mode', 'new file mode']
    for token in spec_tokens:
        if diff.find(token) != -1:
            return True
    return False


class CmgDatasetReaderV2(DatasetReader):
    def __init__(self):
        super().__init__()

    def text_to_instance(self, diff, msg) -> Instance:
        raise NotImplementedError

    def _read(self, data_paths) -> Iterable[Instance]:
        diff_data_path, msg_data_path = data_paths
        diffs = load_data_lines(diff_data_path)
        msgs = load_data_lines(msg_data_path)
        total_diff_loaded = 0

        for diff, msg in tqdm(zip(diffs, msgs), total=len(diffs)):
            if not check_skip_diff_only_complete_code_change(diff):
                total_diff_loaded += 1
                yield self.text_to_instance(diff, msg)

        mylogger.info('CmgDatasetReaderV2._read',
                      f'Total {total_diff_loaded} instances loaded')


@DatasetReader.register("cmg_imp_fix_v2")
class CmgImpFixDatasetReaderV2(CmgDatasetReaderV2):
    def __init__(
        self,
        code_indexers: TokenIndexer,
        code_tokenizer: Optional[Tokenizer] = None,
        code_max_tokens: int = -1,
        msg_max_tokens: int = -1,
        use_op_seq: bool = False,
        use_op_mask: bool = False,
        op_mask_attend_first_token: bool = True,
        code_lower: bool = True,
        msg_lower: bool = True,
        code_align: bool = False,                   # Deprecated
        code_namespace: str = "code_tokens",
        msg_namespace: str = "msg_tokens",
        op_namespace: str = "op_tokens",            # Deprecated
        include_new_del_file_mode: bool = False,    # Deprecated
        line_separator: str = ' ',
        line_extractor_version: str = 'v1',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexers} # or {"tokens": SingleIdTokenIndexer()}
        self.msg_tokenizer = WhitespaceTokenizer()
        self.msg_token_indexers = { msg_namespace: SingleIdTokenIndexer(namespace=msg_namespace) }
        self.op_token_indexer ={ op_namespace: SingleIdTokenIndexer(namespace=op_namespace) }
        self.code_max_tokens = code_max_tokens
        self.msg_max_tokens = msg_max_tokens
        self.use_op_seq = use_op_seq
        self.use_op_mask = use_op_mask
        self.include_new_del_file_mode = include_new_del_file_mode
        self.line_separator = line_separator
        self.op_mask_attend_first_token = op_mask_attend_first_token
        self.differ = difflib.Differ()

        self.code_lower = code_lower
        self.msg_lower = msg_lower
        self.code_align = code_align
        self.line_extractor = get_line_extractor(line_extractor_version)


    def _make_op_mask_tensor_field(self,
                                   tensor: torch.Tensor,
                                   max_len: int=-1):
        end_idx = len(tensor) if max_len == -1 else max_len
        tensor_field = TensorField(tensor[:end_idx])
        return tensor_field

    def tokenize_code(self, code: str) -> List[str]:
        if self.code_tokenizer is None:
            return code.split(' ')  # WhiteSpaceTokenizer in default
        else:
            tokens = self.code_tokenizer.tokenize(code)
            return [token.text for token in tokens]  # fetch text only

    def align_add_del_sequence(self,
                               add_seq: List[str],
                               del_seq: List[str]):
        aligned_diff_seq = construct_diff_sequence(del_seq, add_seq)  # delete lines are ahead input

        aligned_del_tokens = []
        aligned_add_tokens = []
        aligned_ops = []
        for del_token, add_token, op in aligned_diff_seq:
            aligned_del_tokens.append(del_token)
            aligned_add_tokens.append(add_token)
            aligned_ops.append(op)

        return aligned_add_tokens, aligned_del_tokens, aligned_ops


    def text_to_instance(self, diff: str, msg: str) -> Instance:
        add_lines, del_lines = self.line_extractor(diff, line_separator=self.line_separator)

        if self.code_lower:
            add_lines, del_lines = add_lines.lower(), del_lines.lower()
        if self.msg_lower:
            msg = msg.lower()
        add_seq = self.tokenize_code(add_lines)
        del_seq = self.tokenize_code(del_lines)

        fields = {}
        if self.code_align:
            add_seq, del_seq, op_lines = self.align_add_del_sequence(add_seq, del_seq)
            if self.use_op_seq:
                op_tokens = _make_token_list(op_lines, self.code_max_tokens)
                diff_op_field = TextField(op_tokens, self.op_token_indexer)
                fields['diff_op'] = diff_op_field

        add_tokens = _make_token_list(add_seq, self.code_max_tokens)
        del_tokens = _make_token_list(del_seq, self.code_max_tokens)
        diff_add_field = TextField(add_tokens, self.code_token_indexers)
        diff_del_field = TextField(del_tokens, self.code_token_indexers)
        fields['diff_add'] = diff_add_field
        fields['diff_del'] = diff_del_field

        msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg_seq)
        msg_field = make_target_sequence_field(msg_tokens, self.msg_token_indexers)
        fields['msg'] = msg_field

        if self.use_op_mask:
            if self.code_align:
                mylogger.warning('CmgImpFixDatasetReader.text_to_instance',
                                 'Align and op-mask are set to True together, which is not expected usually')
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(
                self.differ, diff_add_field, diff_del_field, self.op_mask_attend_first_token
            )
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)

    def read_raw_data(self, data_paths):
        diff_data_path, msg_data_path = data_paths
        raw_diffs = load_data_lines(diff_data_path)
        raw_msgs = load_data_lines(msg_data_path)
        diffs, msgs = [], []

        for diff, msg in tqdm(zip(raw_diffs, raw_msgs)):
            if not check_skip_diff_only_complete_code_change(diff):
                diffs.append(diff)
                msgs.append(msg)

        print(f'Total {len(diffs)} diffs loaded')
        return diffs, msgs


@DatasetReader.register("cmg_hybrid_fix_v2")
class CmgHybridFixDatasetReaderV2(CmgImpFixDatasetReaderV2):
    """
    This reader will make diff as additional input, compared to ImpFixReader.
    """
    def __init__(
        self,
        code_indexers: TokenIndexer,
        code_tokenizer: Optional[Tokenizer] = None,
        code_max_tokens: int = -1,
        msg_max_tokens: int = -1,
        use_op_seq: bool = False,
        use_op_mask: bool = False,
        op_mask_attend_first_token: bool = True,
        code_lower: bool = True,
        msg_lower: bool = True,
        code_align: bool = False,
        code_namespace: str = "code_tokens",
        msg_namespace: str = "msg_tokens",
        op_namespace: str = "op_tokens",
        include_new_del_file_mode: bool = False,
        line_separator: str = ' ',
        line_extractor_version: str = 'v1',
        **kwargs
    ):
        super().__init__(code_indexers, code_tokenizer,
                         code_max_tokens, msg_max_tokens,
                         use_op_seq,
                         use_op_mask, op_mask_attend_first_token,
                         code_lower, msg_lower,
                         code_align,
                         code_namespace, msg_namespace, op_namespace,
                         include_new_del_file_mode,
                         line_separator,
                         line_extractor_version)

    def text_to_instance(self, diff: str, msg: str) -> Instance:
        add_lines, del_lines = self.line_extractor(diff, line_separator=self.line_separator)

        if self.code_lower:
            add_lines, del_lines = add_lines.lower(), del_lines.lower()
        if self.msg_lower:
            msg = msg.lower()
        add_seq = self.tokenize_code(add_lines)
        del_seq = self.tokenize_code(del_lines)

        diff_seq = self.tokenize_code(diff)
        diff_tokens = _make_token_list(diff_seq, self.code_max_tokens)
        diff_data_field = TextField(diff_tokens, self.code_token_indexers)

        fields = {
            'diff': diff_data_field
        }
        if self.code_align:
            add_seq, del_seq, op_lines = self.align_add_del_sequence(add_seq, del_seq)
            if self.use_op_seq:
                op_tokens = _make_token_list(op_lines, self.code_max_tokens)
                diff_op_field = TextField(op_tokens, self.op_token_indexer)
                fields['diff_op'] = diff_op_field

        add_tokens = _make_token_list(add_seq, self.code_max_tokens)
        del_tokens = _make_token_list(del_seq, self.code_max_tokens)
        diff_add_field = TextField(add_tokens, self.code_token_indexers)
        diff_del_field = TextField(del_tokens, self.code_token_indexers)
        fields['diff_add'] = diff_add_field
        fields['diff_del'] = diff_del_field

        msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg_seq)
        msg_field = make_target_sequence_field(msg_tokens, self.msg_token_indexers)
        fields['msg'] = msg_field

        if self.use_op_mask:
            if self.code_align:
                mylogger.warning('CmgImpFixDatasetReader.text_to_instance',
                                 'Align and op-mask are set to True together, which is not expected usually')
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(
                self.differ, diff_add_field, diff_del_field, self.op_mask_attend_first_token
            )
            assert len(add_op_mask_field.tensor) == len(diff_add_field)
            assert len(del_op_mask_field.tensor) == len(diff_del_field)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)


@DatasetReader.register("cmg_fix_line_align_v2")
class CmgImpFixLineAlignDatasetReaderV2(CmgDatasetReaderV2):
    def __init__(self,
                 # Because max_length of line tokenization and diff tokenization
                 # are different, thus using two tokenizers
                 line_code_tokenizer: Tokenizer,
                 diff_code_tokenizer: Tokenizer,
                 code_indexers: TokenIndexer,
                 line_max_tokens: int = 64,
                 diff_max_tokens: int = 256,
                 max_lines: int = 64,
                 msg_max_tokens: int = -1,
                 use_diff_as_input: bool = True,
                 use_op_mask: bool = False,
                 op_mask_attend_first_token: bool = True,
                 code_lower: bool = True,
                 msg_lower: bool = True,
                 code_namespace: str = "code_tokens",
                 msg_namespace: str = "msg_tokens",
                 line_align_separator: Optional[str] = None,
                 line_separator: Optional[str] = ' ',
                 empty_line_placeholder: Optional[str] = None,
                 insert_empty_line_placeholder: bool = False,
                 keep_tokenizer_head_tail_token: bool = False,
                 jointly_align_add_del_lines: bool = False,
                 align_equal_lines: bool = False,
                 line_extractor_version: str = 'v1',
                 set_first_line_align_equal: bool = False,
                 **kwargs):
        super().__init__()

        self.line_code_tokenizer = line_code_tokenizer
        self.diff_code_tokenizer = diff_code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexers} # or {"tokens": SingleIdTokenIndexer()}
        self.msg_tokenizer = WhitespaceTokenizer()
        self.msg_token_indexers = { msg_namespace: SingleIdTokenIndexer(namespace=msg_namespace, lowercase_tokens=True) }

        # Num configuration
        self.line_max_tokens = line_max_tokens
        self.diff_max_tokens = diff_max_tokens
        self.max_lines = max_lines
        self.msg_max_tokens = msg_max_tokens

        # Op-mask configuration
        self.use_op_mask = use_op_mask
        self.op_mask_attend_first_token = op_mask_attend_first_token

        # Common configuration
        self.use_diff_as_input = use_diff_as_input
        self.code_namespace = code_namespace
        self.msg_namespace = msg_namespace
        self.code_lower = code_lower
        self.msg_lower = msg_lower
        self.line_separator = line_separator
        self.line_align_separator = line_align_separator
        self.line_extractor = get_line_extractor(line_extractor_version)

        # Line-align configuration
        self.empty_line_placeholder = empty_line_placeholder
        self.insert_empty_line_placeholder = insert_empty_line_placeholder
        self.keep_tokenizer_head_tail_token = keep_tokenizer_head_tail_token
        self.line_align_func = align_lines_in_diff_jointly if jointly_align_add_del_lines else \
                               align_lines_in_diff_two_streams
        self.align_equal_lines = align_equal_lines
        self.set_first_line_align_equal = set_first_line_align_equal
        self.differ = difflib.Differ()

        self.lens = []


    def line_align_helper(self, add_lines, del_lines):
        diff = [
            {
                'added_code': add_lines,
                'removed_code': del_lines
            }
        ]
        return self.line_align_func(diff,
                                    self.line_code_tokenizer,
                                    self.line_align_separator,
                                    self.empty_line_placeholder,
                                    self.line_max_tokens, self.diff_max_tokens, self.max_lines,
                                    self.insert_empty_line_placeholder,
                                    self.keep_tokenizer_head_tail_token,
                                    self.align_equal_lines,
                                    self.set_first_line_align_equal)


    def text_to_instance(self, diff: str, msg: str) -> Instance:
        add_lines, del_lines = self.line_extractor(diff, line_separator=self.line_separator, join=False)

        if self.code_lower:
            add_lines, del_lines = [l.lower() for l in add_lines], [l.lower() for l in del_lines]
            diff = diff.lower()
        if self.msg_lower:
            msg = msg.lower()


        aligned_diff = self.line_align_helper(add_lines, del_lines)
        diff_add_field = TextField(aligned_diff['add'], self.code_token_indexers)
        diff_del_field = TextField(aligned_diff['del'], self.code_token_indexers)
        add_line_idx_field = TensorField(aligned_diff['add_line_idx'])
        del_line_idx_field = TensorField(aligned_diff['del_line_idx'])
        fields = {
            'diff_add': diff_add_field,
            'diff_del': diff_del_field,
            'add_line_idx': add_line_idx_field,
            'del_line_idx': del_line_idx_field,
        }

        if self.use_diff_as_input:
            diff_tokens = self.diff_code_tokenizer.tokenize(diff)
            assert len(diff_tokens) <= self.diff_max_tokens
            diff_field = TextField(diff_tokens, self.code_token_indexers)
            fields['diff'] = diff_field

        msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg_seq)
        msg_field = make_target_sequence_field(msg_tokens, self.msg_token_indexers)
        fields['msg'] = msg_field

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(
                self.differ, diff_add_field, diff_del_field, self.op_mask_attend_first_token
            )
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field
            assert len(add_op_mask_field) == len(diff_add_field)
            assert len(del_op_mask_field) == len(diff_del_field)

        return Instance(fields)