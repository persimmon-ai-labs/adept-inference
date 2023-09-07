# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron tokenizers."""

from abc import ABCMeta
from abc import abstractmethod

from typing import Dict, Any, List, Optional
import sentencepiece as spm  # type: ignore


class AbstractTokenizer(metaclass=ABCMeta):
    """Abstract class for tokenizer."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct tokens in the vocabulary."""
        pass

    @property
    @abstractmethod
    def vocab(self) -> Dict[str, int]:
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self) -> Dict[int, str]:
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize the text."""
        pass

    def encode(self, text: str) -> List[int]:
        """Encode the text."""
        return self.tokenize(text)

    def __call__(self, text: str) -> List[int]:
        """Syntactic sugar for tokenize."""
        return self.tokenize(text)

    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """Transform tokens back to a string."""
        pass

    def decode(self, ids: List[int]) -> str:
        """Decode the ids."""
        return self.detokenize(ids)

    @property
    def cls(self) -> int:
        raise NotImplementedError(f"CLS is not provided for {self.name} tokenizer")

    @property
    def sep(self) -> int:
        raise NotImplementedError(f"SEP is not provided for {self.name} tokenizer")

    @property
    def pad(self) -> int:
        raise NotImplementedError(f"PAD is not provided for {self.name} tokenizer")

    @property
    def eod(self) -> int:
        raise NotImplementedError(f"EOD is not provided for {self.name} tokenizer")

    @property
    def eod_token_text(self) -> str:
        """The EOD token string."""
        return self.decode([self.eod])

    @property
    def mask(self) -> int:
        """Get the mask token."""
        raise NotImplementedError(f"MASK is not provided for {self.name} tokenizer")

    @property
    def pad_token(self) -> str:
        """Get the pad token."""
        raise NotImplementedError

    @property
    def max_len_single_sentence(self) -> int:
        """Get the max length of a single sentence."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Get the length of the tokenizer."""
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        """Get the id of the EOS token."""
        raise NotImplementedError

    @property
    def eos_token(self) -> str:
        """Get the EOS token."""
        raise NotImplementedError


class _SentencePieceTokenizer(AbstractTokenizer):
    """Sentece piece tokenizer."""

    def __init__(self, model_file: str) -> None:
        name = "Sentence Piece"
        super().__init__(name)

        # no-member is firing here but the code works fine!
        # pylint: disable=no-member
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(model_file)
        self._vocab_size = self._tokenizer.vocab_size()
        self._tokens = [self._tokenizer.id_to_piece(t) for t in range(self.vocab_size)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}
        self._eod_id = None
        # look for end of document id
        for idx, token in enumerate(self._tokens):
            if token == "|ENDOFTEXT|":
                self._eod_id = idx
                break
        if self._eod_id is None:
            self._eod_id = self._tokenizer.eos_id()
        assert self._eod_id is not None

    @property
    def tokenizer(self) -> spm.SentencePieceProcessor:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def inv_vocab(self):  # type: ignore
        return self._tokens

    def tokenize(self, text: str):  # type: ignore
        # pylint: disable=bare-except, no-member
        try:
            tokenized = self._tokenizer.encode_as_ids(text)
        except:
            tokenized = None
        return tokenized

    def pieces(self, text: str) -> List[str]:
        # pylint: disable=no-member
        pieces: List[str] = self._tokenizer.encode_as_pieces(text)
        return pieces

    def detokenize(self, token_ids: List[int]) -> str:
        # pylint: disable=no-member
        return self._tokenizer.decode_ids(token_ids)  # type: ignore

    @property
    def eod(self) -> int:
        return self._eod_id  # type: ignore

    @property
    def eos_token_id(self) -> int:
        """Id of the end of sentence token in the vocabulary."""
        eos_id: int = self._eod_id  # type: ignore
        return eos_id


def megatron_initialize_tokenizer(
    tokenizer_type: str,
    sp_model_file: Optional[str] = None,
) -> AbstractTokenizer:
    """Initialize the tokenizer."""
    if tokenizer_type == "SentencePiece":
        assert sp_model_file is not None
        tokenizer = _SentencePieceTokenizer(sp_model_file)
    else:
        raise NotImplementedError(f"{tokenizer_type} tokenizer is not implemented.")
    return tokenizer


def _vocab_size_with_padding(orig_vocab_size: int, args: Any) -> int:
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            f" > padded vocab (sz: {orig_vocab_size}) w/ {after - orig_vocab_size} dummy toks (new sz: {after})",
            flush=True,
        )
    return after


def build_tokenizer(args: Any) -> AbstractTokenizer:
    """Initialize tokenizer."""
    if args.rank == 0:
        print(
            f"> building {args.tokenizer_type} tokenizer ...",
            flush=True,
        )

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == "SentencePiece":
        assert args.sp_model_file is not None
        tokenizer = _SentencePieceTokenizer(f"{args.sp_model_file}")
    else:
        raise NotImplementedError(
            f"{args.tokenizer_type} tokenizer is not implemented."
        )

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer
