# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    if args.tokenizer_type != 'FalconTokenizer':
        assert args.vocab_file is not None

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        tokenizer = _SentencePieceTokenizer(args.vocab_file, vocab_extra_ids=args.vocab_extra_ids, 
                                            vocab_extra_ids_list=args.vocab_extra_ids_list, new_tokens=args.new_tokens)
    elif args.tokenizer_type == 'FalconTokenizer':
        tokenizer = _FalconTokenizer(vocab_extra_ids_list=args.vocab_extra_ids_list, new_tokens=args.new_tokens)
    elif args.tokenizer_type == 'Tiktoken':
        tokenizer = _Tiktoken(args.vocab_file, vocab_extra_ids_list=args.vocab_extra_ids_list, new_tokens=args.new_tokens)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    # args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,args)
    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        SPECIAL_TOKENS = {'eos_token': '[EOS]',
                          'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """ Beginning of sentence token id """
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token id """
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = 'GPT2 BPE'
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class _FalconTokenizer(AbstractTokenizer):
    """Wrapper of huggingface tokenizer."""

    def __init__(self, vocab_extra_ids_list=None, new_tokens=True):
        name = 'FalconTokenizer'
        super().__init__(name)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-40b')
        self._eod = self.tokenizer.vocab['<|endoftext|>']

        if vocab_extra_ids_list and new_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.tokenizer.additional_special_tokens + vocab_extra_ids_list.split(",")})

        self._inv_vocab = {idx: token for token, idx in self.tokenizer.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def eod(self):
        return self._eod


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0, vocab_extra_ids_list=None, new_tokens=True):
        name = 'SentencePieceTokenizer'
        super().__init__(name)

        import sentencepiece
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._initalize(vocab_extra_ids, vocab_extra_ids_list, new_tokens)

    def _initalize(self, vocab_extra_ids, vocab_extra_ids_list, new_tokens):
        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for i in range(len(self._tokenizer)):
            t = self._tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

        def _add_special_token(t, force=False):
            if t not in self.vocab and not new_tokens and not force:
                return
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab.get('<CLS>')
        _add_special_token('<SEP>')
        self._sep_id = self._vocab.get('<SEP>')
        _add_special_token('<EOD>')
        self._eod_id = self._vocab.get('<EOD>')
        _add_special_token('<MASK>')
        self._mask_id = self._vocab.get('<MASK>')

        pad_id = self._tokenizer.pad_id()
        try:
            pad_token = self._tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab.get(pad_token)

        bos_id = self._tokenizer.bos_id()
        try:
            bos_token = self._tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab.get(bos_token)

        eos_id = self._tokenizer.eos_id()
        try:
            eos_token = self._tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab.get(eos_token)

        if not new_tokens:
            # default to eos
            self._pad_id = self._eos_id

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t, force=True)
            self._t5_tokens += [t]
        if vocab_extra_ids_list:
            for t in vocab_extra_ids_list.split(","):
                _add_special_token(t, force=True)
        print("Special tokens: {}".format(self._special_tokens))

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1
        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id  # in case noe eod we can patch this up with an eos

    @property
    def eos_token_id(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path

class _Tiktoken(AbstractTokenizer):
    """Tiktoken-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0, vocab_extra_ids_list=None, new_tokens=True):
        name = 'Tiktoken'
        super().__init__(name)

        import tiktoken
        self.model_path = model_file
        self._tokenizer = load_tiktoken_bpe(self.model_path)
        self.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self.num_base_tokens = len(self._tokenizer)
        self.num_reserved_special_tokens = 256
        
        special_tokens = [
                    "<|begin_of_text|>",
                    "<|end_of_text|>",
                    "<CLS>",
                    "<SEP>",
                    "<EOD>",
                    "<MASK>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<PAD>",
                    "<|eot_id|>",  # end of turn
                ]
        if vocab_extra_ids_list:
            for t in vocab_extra_ids_list.split(","):
                special_tokens += [t]
            
        special_tokens += [
                f"<|reserved_special_token_{i}|>"
                for i in range(0, self.num_reserved_special_tokens - len(special_tokens))
            ]
        
        self._special_tokens = {
            token: self.num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self._inv_special_tokens = {
            self.num_base_tokens + i : token for i, token in enumerate(special_tokens)
        }
        # print("Special tokens: {}".format(self._special_tokens))
        
        self.model = tiktoken.Encoding(
            name=Path(self.model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=self._tokenizer,
            special_tokens=self._special_tokens,
        )
        
        self._initalize(None, None, None)
        for token, id in self._special_tokens.items():
            self._vocab[token] = id
            self._inv_vocab[id] = token
        self._cls_id = self._vocab.get('<CLS>')
        self._sep_id = self._vocab.get('<SEP>')
        self._eod_id = self._vocab.get('<EOD>')
        self._mask_id = self._vocab.get('<MASK>')
        self._pad_id = self._vocab.get('<PAD>')
        self._bos_id = self._vocab.get('<|begin_of_text|>')
        self._eos_id = self._vocab.get('<|end_of_text|>')
        print("DEBUG LINE!!!!!", self._vocab["<|im_start|>"])

    def _initalize(self, vocab_extra_ids, vocab_extra_ids_list, new_tokens):
        self._vocab = {}
        self._inv_vocab = {}

        self._t5_tokens = []

        for token,id in self._tokenizer.items():
            self._inv_vocab[id] = token
            self._vocab[token] = id

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int):
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]
        
    # From:
    # https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
    def tokenize(self, s, allowed_special= set(), disallowed_special= ()):
        # print(s)
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        # if bos:
            # t.insert(0, self.bos_id)
        # if eos:
            # t.append(self.eos_id)
        return t

    # From:
    # https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
    def detokenize(self, ids):
        return self.model.decode(ids)

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id  # in case noe eod we can patch this up with an eos

    @property
    def eos_token_id(self):
        if self._eod_id is not None:
            return self._eod_id
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

