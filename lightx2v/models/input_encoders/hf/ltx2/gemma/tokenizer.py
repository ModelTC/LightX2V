from transformers import AutoTokenizer, PreTrainedTokenizerBase


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 256):
        """
        Initialize the tokenizer.
        Args:
            tokenizer_path (str): Path to the pretrained tokenizer files or model directory.
            max_length (int, optional): Max sequence length for encoding. Defaults to 256.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, model_max_length=max_length)
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> "LTXVGemmaTokenizer":
        """Wrap an already constructed tokenizer.

        LTX-2.5 stores its tokenizer inside the text-encoder safetensors file,
        so there is no directory that can be passed to ``AutoTokenizer``.  The
        regular path-based constructor remains unchanged for LTX-2/LTX-2.3.
        """
        obj = cls.__new__(cls)
        obj.tokenizer = tokenizer
        obj.tokenizer.model_max_length = max_length
        obj.tokenizer.padding_side = "left"
        if obj.tokenizer.pad_token is None:
            obj.tokenizer.pad_token = obj.tokenizer.eos_token
        obj.max_length = max_length
        return obj

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Gemma tokenizer is missing bos_token_id")
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0].tolist()
        # Gemma 3 tokenizers already add BOS; the packed Gemma 4 tokenizer
        # does not.  Normalize both families to the source pipeline contract.
        if not input_ids or input_ids[0] != bos_id:
            input_ids = [bos_id, *input_ids][: self.max_length]
        encoded = self.tokenizer.pad(
            {"input_ids": [input_ids]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [(token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out
