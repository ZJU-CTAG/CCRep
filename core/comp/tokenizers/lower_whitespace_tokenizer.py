from typing import List, Dict, Any

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("lower_whitespace")
class LowerWhitespaceTokenizer(Tokenizer):
    """
    A `Tokenizer` that is almost the same as WhitespaceTokenizer, except for using
    .lower() to do some simple word stemming before making a word token.
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t.lower()) for t in text.split()]

    def _to_params(self) -> Dict[str, Any]:
        return {"type": "lower_whitespace"}
