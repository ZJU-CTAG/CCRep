import difflib
from typing import Iterable, Dict, List, Optional, Tuple

from tqdm import tqdm

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from utils.data_utils.line_align_utils import align_lines_in_diff_jointly, align_lines_in_diff_two_streams
from utils.data_utils.op_mask_utils import diff_and_gen_op_mask
from utils.diff import remake_diff_from_hunks
from utils.file import read_dumped


@DatasetReader.register("apca_imp_base")
class ImpBaseAPCADatasetReader(DatasetReader):
    '''
    Implicit code change dataset reader for apca task.
    An instance must contain 'add', 'del' and 'label' fields.
    'Op' field is optional for aligned sequences, and it can
    be None for unaligned sequences.
    '''

    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            max_tokens: int = 128,  # only valid for 'flat' diff structure
            empty_code_token: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {"code_tokens": code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.empty_code_token = empty_code_token
        self.differ = difflib.Differ()


    def make_diff_field_from_hunks(self, diff_hunks):
        diff_str = remake_diff_from_hunks(diff_hunks, self.line_joiner)
        # Empty diff lines
        if diff_str == '':
            diff_tokens = self.get_token_list_for_empty_code()
        else:
            diff_tokens = self.code_tokenizer.tokenize(diff_str)

        return TextField(diff_tokens[:self.max_tokens], self.code_token_indexers)


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


    def text_to_instance(self, patch: Dict) -> Instance:
        raise NotImplementedError


    def _read(self, file_path: str) -> Iterable[Instance]:
        patches = read_dumped(file_path)

        for patch in tqdm(patches, total=len(patches)):
            yield self.text_to_instance(patch)


@DatasetReader.register("apca_imp_flat")
class ImpFlatAPCAatasetReader(ImpBaseAPCADatasetReader):
    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            max_tokens: int = None,
            use_op_mask: bool = False,
            op_mask_attend_first_token: bool = True,
            lower_code: bool = False,
            hunk_separator: Optional[str] = None,
            line_joiner: str = ' ',
            empty_code_token: Optional[str] = None,
            use_diff: bool = False,
            **kwargs
    ):
        super().__init__(code_indexer, code_tokenizer, max_tokens,
                         empty_code_token,
                         **kwargs)
        self.lower_code = lower_code
        self.hunk_separator = hunk_separator
        self.line_joiner = line_joiner
        self.tokenized_diff = []
        self.token_lens = []
        self.use_op_mask = use_op_mask
        self.op_mask_attend_first_token = op_mask_attend_first_token
        self.use_diff = use_diff


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


    def text_to_instance(self, patch: Dict) -> Instance:
        diff = patch.get('diff')
        label = patch.get('label')

        add_diff, del_diff = self.extract_add_del_code_hierarchy(diff)
        diff_add_field = self._flatten_and_make_text_field(add_diff)
        diff_del_field = self._flatten_and_make_text_field(del_diff)

        fields = {
            'diff_add': diff_add_field,
            'diff_del': diff_del_field,
        }
        if self.use_diff:
            diff_field = self.make_diff_field_from_hunks(diff)
            fields['diff'] = diff_field

        # make label as a field only wh  en it is available
        if label is not None:
            # Make label a numerical field
            fields['label'] = LabelField(int(label), skip_indexing=True)

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(self.differ,
                                                                        diff_add_field, diff_del_field,
                                                                        self.op_mask_attend_first_token)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)


@DatasetReader.register("apca_imp_flat_line_align")
class ImpFlatLineAlignAPCADatasetReader(ImpBaseAPCADatasetReader):
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
            lower_code: bool = False,
            keep_tokenizer_head_tail_token: bool = False,
            jointly_align_add_del_lines: bool = True,
            align_equal_lines: bool = True,
            **kwargs
    ):
        super().__init__(code_indexer, code_tokenizer, diff_max_tokens,
                         None,
                         **kwargs)
        self.lower_code = lower_code

        self.line_separator = line_separator
        self.line_max_tokens = line_max_tokens
        self.diff_max_tokens = diff_max_tokens
        self.max_lines = max_lines
        self.empty_line_placeholder = empty_line_placeholder
        self.insert_empty_line_placeholder = insert_empty_line_placeholder
        self.keep_tokenizer_head_tail_token = keep_tokenizer_head_tail_token
        self.align_equal_lines = align_equal_lines
        self.use_op_mask = use_op_mask
        self.op_mask_attend_first_token = op_mask_attend_first_token
        self.line_align_func = align_lines_in_diff_jointly if jointly_align_add_del_lines else \
                               align_lines_in_diff_two_streams

        self.debug_lens = []


    def text_to_instance(self, patch: Dict) -> Instance:
        diff = patch.get('diff')
        label = patch.get('label')

        aligned_diff = self.line_align_func(diff, self.code_tokenizer,
                                            self.line_separator,
                                            self.empty_line_placeholder,
                                            self.line_max_tokens,
                                            self.diff_max_tokens,
                                            self.max_lines,
                                            self.insert_empty_line_placeholder,
                                            self.keep_tokenizer_head_tail_token,
                                            self.align_equal_lines)
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

        # make label as a field only when it is available
        if label is not None:
            # Make label a numerical field
            fields['label'] = LabelField(int(label), skip_indexing=True)

        if self.use_op_mask:
            add_op_mask_field, del_op_mask_field = diff_and_gen_op_mask(self.differ,
                                                                        diff_add_field, diff_del_field,
                                                                        self.op_mask_attend_first_token)
            fields['add_op_mask'] = add_op_mask_field
            fields['del_op_mask'] = del_op_mask_field

        return Instance(fields)