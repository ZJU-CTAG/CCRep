from typing import List, Dict, Any, Optional

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("truncate_whitespace")
class TruncateWhitespaceTokenizer(Tokenizer):

    def __init__(self,
                 max_length: int = -1,
                 start_token: Optional[str] = None):
        self.max_length = max_length if max_length != -1 else 1 << 60
        self.start_token = start_token

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = [Token(t) for t in text.split()[:self.max_length]]
        if self.start_token is not None:
            tokens.insert(0, Token(self.start_token))
        return tokens

    def _to_params(self) -> Dict[str, Any]:
        return {"type": "truncate_whitespace"}
