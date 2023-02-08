####################################################################
#
#   Reader for FIRA data (json), from git-diff.
#
####################################################################

import difflib
from typing import Iterable, Dict, List, Optional

import torch
from tqdm import tqdm

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TokenIndexer, Tokenizer, Field, Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from core.comp.tokenizers.white_space_truncate import TruncateWhitespaceTokenizer
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.extract_utils import get_line_extractor
from utils.data_utils.cmg_data_utils import make_target_sequence_field
from utils.data_utils.git_diff_utils import get_clean_diff_meta_data_func
from utils.data_utils.line_align_utils import align_lines_in_diff_jointly, align_lines_in_diff_two_streams
from utils.data_utils.placeholder_utils import replace_var_with_placeholder
from utils.file import load_json, read_dumped
from utils.data_utils.op_mask_utils import diff_and_gen_op_mask
from utils.data_utils.lemmatization import apply_fira_lemmatization

def load_data_lines(path):
    """load lines from a file"""
    lines = load_json(path)
    return lines


def _make_token_list(str_list: Optional[List[str]],
                     max_length: int = -1) -> Optional[List[Token]]:
    ret = [Token(token) for i, token in enumerate(str_list) if max_length == -1 or i < max_length]
    return ret


class CmgDatasetReaderFira(DatasetReader):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

    def text_to_instance(self, diff: Dict, msg: str) -> Instance:
        raise NotImplementedError

    def _read(self, data_paths) -> Iterable[Instance]:
        diff_data_path, msg_data_path = data_paths
        diffs = load_data_lines(diff_data_path)
        msgs = load_data_lines(msg_data_path)
        total_diff_loaded = 0

        # if 'train' in diff_data_path or 'validate' in diff_data_path:
        if self.debug:
            diffs = diffs[:1000]
            msgs = msgs[:1000]

        for diff, msg in tqdm(zip(diffs, msgs), total=len(diffs)):
            total_diff_loaded += 1
            yield self.text_to_instance(diff, msg)

        mylogger.info('CmgDatasetReaderFira._read',
                      f'Total {total_diff_loaded} instances loaded')


@DatasetReader.register("cmg_imp_fix_fira")
class CmgImpFixDatasetReaderFira(CmgDatasetReaderFira):
    def __init__(
        self,
        code_indexers: TokenIndexer,
        code_tokenizer: Optional[Tokenizer] = None,
        msg_tokenizer: Optional[Tokenizer] = None,
        msg_token_indexer: Optional[TokenIndexer] = None,
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
        include_new_del_file_mode: bool = False,    # Deprecated
        line_separator: str = ' ',
        line_extractor_version: str = 'fira_v1',
        add_msg_start_end_tokens: bool = True,
        start_token: Optional[str] = START_SYMBOL,
        end_token: Optional[str] = END_SYMBOL,
        use_fira_lemmatization: bool = False,
        use_identifier_placeholder: bool = False,
        keep_identifer_before_placeholder: bool = True,
        minimum_replace_length: int = 2,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(debug)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexers} # or {"tokens": SingleIdTokenIndexer()}
        self.msg_tokenizer = msg_tokenizer if msg_tokenizer is not None else TruncateWhitespaceTokenizer(msg_max_tokens)
        self.msg_token_indexers = { msg_namespace: msg_token_indexer
                                                   if msg_token_indexer is not None else
                                                   SingleIdTokenIndexer(namespace=msg_namespace, lowercase_tokens=True) }
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
        self.add_msg_start_end_tokens = add_msg_start_end_tokens
        self.start_token = start_token
        self.end_token = end_token
        self.use_fira_lemmatization = use_fira_lemmatization
        self.use_identifier_placeholder = use_identifier_placeholder
        self.keep_identifer_before_placeholder = keep_identifer_before_placeholder
        self.minimum_replace_length = minimum_replace_length


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


    def text_to_instance(self, diff: Dict, msg: str) -> Instance:
        diff_str = diff['diff']
        add_lines, del_lines = self.line_extractor(diff_str, line_separator=self.line_separator)
        if self.use_identifier_placeholder:
            diff_var_mapping = diff['vars']
            add_lines, _ = replace_var_with_placeholder(add_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            del_lines, _ = replace_var_with_placeholder(del_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            diff_str, _ = replace_var_with_placeholder(diff_str, diff_var_mapping, keep_mapped_var=self.keep_identifer_before_placeholder, minimum_replace_length=self.minimum_replace_length)
            # Msg should always not keep the original identifiers
            msg, _ = replace_var_with_placeholder(msg, diff_var_mapping, keep_mapped_var=False, minimum_replace_length=self.minimum_replace_length)

        if self.code_lower:
            add_lines, del_lines = add_lines.lower(), del_lines.lower()
        if self.msg_lower:
            msg = msg.lower()
        if self.use_fira_lemmatization:
            msg = apply_fira_lemmatization(msg)
        add_seq = self.tokenize_code(add_lines)
        del_seq = self.tokenize_code(del_lines)

        fields = {}
        if self.code_align:
            raise NotImplementedError

        add_tokens = _make_token_list(add_seq, self.code_max_tokens)
        del_tokens = _make_token_list(del_seq, self.code_max_tokens)
        diff_add_field = TextField(add_tokens, self.code_token_indexers)
        diff_del_field = TextField(del_tokens, self.code_token_indexers)
        fields['diff_add'] = diff_add_field
        fields['diff_del'] = diff_del_field

        # msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg)
        msg_field = make_target_sequence_field(msg_tokens,
                                               self.msg_token_indexers,
                                               self.add_msg_start_end_tokens,
                                               self.start_token, self.end_token)
        fields['msg'] = msg_field
        if self.use_identifier_placeholder:
            fields['meta_data'] = MetadataField({'vars': diff['vars']})

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


@DatasetReader.register("cmg_hybrid_fix_fira")
class CmgHybridFixDatasetReaderFira(CmgImpFixDatasetReaderFira):
    def __init__(
        self,
        code_indexers: TokenIndexer,
        code_tokenizer: Optional[Tokenizer] = None,
        msg_tokenizer: Optional[Tokenizer] = None,
        msg_token_indexer: Optional[TokenIndexer] = None,
        code_max_tokens: int = -1,
        msg_max_tokens: int = -1,
        use_op_seq: bool = False,                   # Deprecated
        use_op_mask: bool = False,
        op_mask_attend_first_token: bool = True,
        code_lower: bool = True,
        msg_lower: bool = True,
        code_align: bool = False,
        code_namespace: str = "code_tokens",
        msg_namespace: str = "msg_tokens",
        op_namespace: str = "op_tokens",            # Deprecated
        include_new_del_file_mode: bool = False,    # Deprecated
        line_separator: str = ' ',
        line_extractor_version: str = 'fira_v1',
        clean_raw_diff: bool = True,
        clean_raw_diff_version: str = 'v1',
        add_msg_start_end_tokens: bool = True,
        start_token: Optional[str] = START_SYMBOL,
        end_token: Optional[str] = END_SYMBOL,
        use_fira_lemmatization: bool = False,
        use_identifier_placeholder: bool = False,
        keep_identifer_before_placeholder: bool = True,
        minimum_replace_length: int = 2,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(code_indexers, code_tokenizer,
                         msg_tokenizer, msg_token_indexer,
                         code_max_tokens, msg_max_tokens,
                         use_op_seq,
                         use_op_mask, op_mask_attend_first_token,
                         code_lower, msg_lower,
                         code_align,
                         code_namespace, msg_namespace, op_namespace,
                         include_new_del_file_mode,
                         line_separator,
                         line_extractor_version,
                         add_msg_start_end_tokens,
                         start_token, end_token,
                         use_fira_lemmatization,
                         use_identifier_placeholder, keep_identifer_before_placeholder,
                         minimum_replace_length,
                         debug)
        self.clean_raw_diff = clean_raw_diff
        self.clean_raw_diff_func = get_clean_diff_meta_data_func(clean_raw_diff_version)

    def text_to_instance(self, diff: Dict, msg: str) -> Instance:
        diff_str = diff['diff']
        add_lines, del_lines = self.line_extractor(diff_str, line_separator=self.line_separator)
        if self.clean_raw_diff:
            diff_str = self.clean_raw_diff_func(diff_str, self.line_separator)

        if self.use_identifier_placeholder:
            diff_var_mapping = diff['vars']
            add_lines, _ = replace_var_with_placeholder(add_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            del_lines, _ = replace_var_with_placeholder(del_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            diff_str, _ = replace_var_with_placeholder(diff_str, diff_var_mapping, keep_mapped_var=self.keep_identifer_before_placeholder, minimum_replace_length=self.minimum_replace_length)
            # Msg should always not keep the original identifiers
            msg, _ = replace_var_with_placeholder(msg, diff_var_mapping, keep_mapped_var=False, minimum_replace_length=self.minimum_replace_length)

        if self.code_lower:
            add_lines, del_lines = add_lines.lower(), del_lines.lower()
        if self.msg_lower:
            msg = msg.lower()
        if self.use_fira_lemmatization:
            msg = apply_fira_lemmatization(msg)

        add_seq = self.tokenize_code(add_lines)
        del_seq = self.tokenize_code(del_lines)
        diff_seq = self.tokenize_code(diff_str)
        diff_tokens = _make_token_list(diff_seq, self.code_max_tokens)
        diff_data_field = TextField(diff_tokens, self.code_token_indexers)

        fields = {
            'diff': diff_data_field
        }
        if self.code_align:
            raise NotImplementedError

        add_tokens = _make_token_list(add_seq, self.code_max_tokens)
        del_tokens = _make_token_list(del_seq, self.code_max_tokens)
        diff_add_field = TextField(add_tokens, self.code_token_indexers)
        diff_del_field = TextField(del_tokens, self.code_token_indexers)
        fields['diff_add'] = diff_add_field
        fields['diff_del'] = diff_del_field

        # msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg)
        msg_field = make_target_sequence_field(msg_tokens,
                                               self.msg_token_indexers,
                                               self.add_msg_start_end_tokens,
                                               self.start_token, self.end_token)
        fields['msg'] = msg_field
        if self.use_identifier_placeholder:
            fields['meta_data'] = MetadataField({'vars': diff['vars']})

        if self.use_op_mask:
            if self.code_align:
                mylogger.warning('CmgHybridFixDatasetReaderFira.text_to_instance',
                                 'Align and op-mask are set to True together, which is not expected usually')
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(
                self.differ, diff_add_field, diff_del_field, self.op_mask_attend_first_token
            )
            assert len(add_op_mask_field.tensor) == len(diff_add_field)
            assert len(del_op_mask_field.tensor) == len(diff_del_field)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)


@DatasetReader.register("cmg_fix_line_align_fira")
class CmgImpFixLineAlignDatasetReaderFira(CmgDatasetReaderFira):
    def __init__(self,
                 # Because max_length of line tokenization and diff tokenization
                 # are different, thus using two tokenizers
                 line_code_tokenizer: Tokenizer,
                 diff_code_tokenizer: Tokenizer,
                 code_indexers: TokenIndexer,
                 msg_tokenizer: Optional[Tokenizer] = None,
                 msg_token_indexer: Optional[TokenIndexer] = None,
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
                 line_extractor_version: str = 'fira_v1',
                 set_first_line_align_equal: bool = False,
                 clean_raw_diff: bool = True,
                 clean_raw_diff_version: str = 'v1',
                 add_msg_start_end_tokens: bool = True,
                 start_token: Optional[str] = START_SYMBOL,
                 end_token: Optional[str] = END_SYMBOL,
                 use_fira_lemmatization: bool = False,
                 use_identifier_placeholder: bool = False,
                 keep_identifer_before_placeholder: bool = True,
                 minimum_replace_length: int = 2,
                 debug: bool = False,
                 **kwargs):
        super().__init__(debug)

        self.line_code_tokenizer = line_code_tokenizer
        self.diff_code_tokenizer = diff_code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexers} # or {"tokens": SingleIdTokenIndexer()}
        self.msg_tokenizer = msg_tokenizer if msg_tokenizer is not None else TruncateWhitespaceTokenizer(msg_max_tokens)
        self.msg_token_indexers = { msg_namespace: msg_token_indexer
                                                   if msg_token_indexer is not None else
                                                   SingleIdTokenIndexer(namespace=msg_namespace, lowercase_tokens=True) }

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
        self.clean_raw_diff = clean_raw_diff
        self.clean_raw_diff_func = get_clean_diff_meta_data_func(clean_raw_diff_version)
        self.add_msg_start_end_tokens = add_msg_start_end_tokens
        self.start_token = start_token
        self.end_token = end_token
        self.use_fira_lemmatization = use_fira_lemmatization
        self.use_fira_lemmatization = use_fira_lemmatization
        self.use_identifier_placeholder = use_identifier_placeholder
        self.keep_identifer_before_placeholder = keep_identifer_before_placeholder
        self.minimum_replace_length = minimum_replace_length

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


    def text_to_instance(self, diff: Dict, msg: str) -> Instance:
        diff_str = diff['diff']
        add_lines, del_lines = self.line_extractor(diff_str, line_separator=self.line_separator, join=False)
        if self.clean_raw_diff:
            diff_str = self.clean_raw_diff_func(diff_str, self.line_separator)

        if self.use_identifier_placeholder:
            diff_var_mapping = diff['vars']
            add_lines, _ = replace_var_with_placeholder(add_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            del_lines, _ = replace_var_with_placeholder(del_lines, diff_var_mapping, self.line_separator, self.keep_identifer_before_placeholder, self.minimum_replace_length)
            diff_str, _ = replace_var_with_placeholder(diff_str, diff_var_mapping, keep_mapped_var=self.keep_identifer_before_placeholder, minimum_replace_length=self.minimum_replace_length)
            # Msg should always not keep the original identifiers
            msg, _ = replace_var_with_placeholder(msg, diff_var_mapping, keep_mapped_var=False, minimum_replace_length=self.minimum_replace_length)

        if self.code_lower:
            add_lines, del_lines = [l.lower() for l in add_lines], [l.lower() for l in del_lines]
            diff_str = diff_str.lower()
        if self.msg_lower:
            msg = msg.lower()
        if self.use_fira_lemmatization:
            msg = apply_fira_lemmatization(msg)

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
            diff_tokens = self.diff_code_tokenizer.tokenize(diff_str)
            assert len(diff_tokens) <= self.diff_max_tokens
            diff_field = TextField(diff_tokens, self.code_token_indexers)
            fields['diff'] = diff_field

        # msg_seq = _split_and_truncate(msg, self.msg_max_tokens)
        msg_tokens = self.msg_tokenizer.tokenize(msg)
        msg_field = make_target_sequence_field(msg_tokens,
                                               self.msg_token_indexers,
                                               self.add_msg_start_end_tokens,
                                               self.start_token, self.end_token)
        fields['msg'] = msg_field
        if self.use_identifier_placeholder:
            fields['meta_data'] = MetadataField({'vars': diff['vars']})

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(
                self.differ, diff_add_field, diff_del_field, self.op_mask_attend_first_token
            )
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field
            assert len(add_op_mask_field) == len(diff_add_field)
            assert len(del_op_mask_field) == len(diff_del_field)

        return Instance(fields)