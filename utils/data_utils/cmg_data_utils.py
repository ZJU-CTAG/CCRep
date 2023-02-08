from typing import Iterable, Dict, List, Optional

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token

def make_target_sequence_field(tokenized_sequence: List[Token],
                               token_indexers: Dict[str, TokenIndexer],
                               add_start_end_tokens: bool = True,
                               start_token: Optional[str] = START_SYMBOL,
                               end_token: Optional[str] = END_SYMBOL,
                               ) -> TextField:
    if add_start_end_tokens:
        # add start and end symbol to target sequence
        if start_token is not None:
            tokenized_sequence.insert(0, Token(start_token))
        if end_token is not None:
            tokenized_sequence.append(Token(end_token))
    field = TextField(tokenized_sequence, token_indexers)
    # msg_field.count_vocab_items(self.msg_token_counter)
    return field