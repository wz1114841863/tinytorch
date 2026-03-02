import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


# Constants for memory calculations
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion


class Tokenizer:
    """Base tokenizer class for providing the interface for all tokenziers."""

    def encode(self, text):
        """Converts a string of text into a list of token IDs."""
        raise NotImplementedError(
            "The encode method must be implemented by subclasses."
        )

    def decode(self, tokens):
        """Converts a list of token IDs back into a string of text."""
        raise NotImplementedError(
            "The decode method must be implemented by subclasses."
        )


class CharTokenizer(Tokenizer):
    """A simple character-level tokenizer that maps each unique character to a token ID."""

    def __init__(self, vocab=None):
        """Initializes the tokenizer with a given text to build the vocabulary."""
        if vocab is None:
            vocab = []
        self.vocab = ["<UNK>"] + vocab
        self.vocab_size = len(self.vocab)

        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.unk_id = 0

    def build_vocab(self, corpus):
        """Builds a set of unique characters from the input text."""
        all_chars = set()
        for text in corpus:
            all_chars.update(set(text))

        self.vocab = ["<UNK>"] + sorted(all_chars)
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text into a list of token IDs."""
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs back into a string of text."""
        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, "<UNK>")
            chars.append(char)
        return "".join(chars)


def _count_byte_pairs(word_tokens, word_freq):
    """Counts the frequency of byte pairs in the given word tokens.

    EXAMPLE:
    >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o</w>']}
    >>> word_freq = Counter({"hello": 3})
    >>> counts = _count_byte_pairs(word_tokens, word_freq)
    >>> counts[('h', 'e')]
    3
    """
    pair_freq = Counter()
    for word, freq in zip(word_tokens, word_freq):
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
    return pair_freq


def _merge_pair(word_tokens, pair):
    """Merges a given pair in all word tokens.

    >>> word_tokens = {"hello": ['h', 'e', 'l', 'l', 'o</w>']}
    >>> merged = _merge_pair(word_tokens, ('h', 'e'))
    >>> word_tokens["hello"]
    ['he', 'l', 'l', 'o</w>']
    >>> merged
    'he'
    """
    merged_token = pair[0] + pair[1]

    for word in word_tokens:
        tokens = word_tokens[word]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                # Merge pair
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        word_tokens[word] = new_tokens

    return merged_token


class BPETokenizer(Tokenizer):
    """A Byte Pair Encoding (BPE) tokenizer that builds a vocabulary based on the most frequent byte pairs."""

    def __init__(self, vocab_size: int = 1000):
        """Initializes the BPE tokenizer with a specified vocabulary size."""
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_word_tokens(self, word: str) -> List[str]:
        """Convert word to list of characters with end-of-word marker.

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o</w>']
        """
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += "</w>"  # Mark end of word
        return tokens

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs from word tokens.

        EXAMPLE:
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def train(self, corpus, vocab_size):
        """Train BPE on corpus to learn merge rules."""
        if vocab_size:
            self.vocab_size = vocab_size

        word_freq = Counter(corpus)
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        self.vocab = sorted(vocab)
        if "<UNK>" not in vocab:
            self.vocab = ["<UNK>"] + self.vocab

        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts = _count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = _merge_pair(word_tokens, best_pair)
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned merge rules to token sequence.

        EXAMPLE:
        >>> # After training, merges might be [('h','e'), ('l','l')]
        >>> tokenizer._apply_merges(['h','e','l','l','o</w>'])
        ['he','ll','o</w>']  # Applied both merges
        """
        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge_pair[0]
                    and tokens[i + 1] == merge_pair[1]
                ):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE.

        EXAMPLE:
        >>> tokenizer.encode("hello world")
        [12, 45, 78]  # Token IDs after BPE merging
        """
        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text.

        EXAMPLE:
        >>> tokenizer.decode([12, 45, 78])
        "hello world"  # Reconstructed text
        """
        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, "<UNK>")
            token_strings.append(token)

        # Join and clean up
        text = "".join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace("</w>", " ")

        # Clean up extra spaces
        text = " ".join(text.split())

        return text


def create_tokenizer(
    strategy: str = "char", vocab_size: int = 1000, corpus: List[str] = None
) -> Tokenizer:
    """Factory function to create and train tokenizers.

    EXAMPLE:
    >>> corpus = ["hello world", "test text"]
    >>> tokenizer = create_tokenizer("char", corpus=corpus)
    >>> tokens = tokenizer.encode("hello")
    """

    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'\n"
            f"  ❌ Strategy '{strategy}' is not recognized\n"
            f"  💡 TinyTorch supports 'char' (character-level) and 'bpe' (byte-pair encoding) strategies\n"
            f"  🔧 Use: create_tokenizer('char', corpus=texts) or create_tokenizer('bpe', vocab_size=1000, corpus=texts)"
        )

    return tokenizer


def tokenize_dataset(
    texts: List[str], tokenizer: Tokenizer, max_length: int = None
) -> List[List[int]]:
    """Tokenize a dataset with optional length limits.

    EXAMPLE:
    >>> texts = ["hello world", "tokenize this"]
    >>> tokenizer = CharTokenizer(['h','e','l','o',' ','w','r','d','t','k','n','i','z','s'])
    >>> tokenized = tokenize_dataset(texts, tokenizer, max_length=10)
    >>> all(len(seq) <= 10 for seq in tokenized)
    True
    """
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized


def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """Analyze tokenization statistics.

    EXAMPLE:
    >>> texts = ["hello", "world"]
    >>> tokenizer = CharTokenizer(['h','e','l','o','w','r','d'])
    >>> stats = analyze_tokenization(texts, tokenizer)
    >>> 'vocab_size' in stats and 'avg_sequence_length' in stats
    True
    """
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        "vocab_size": tokenizer.vocab_size,
        "avg_sequence_length": np.mean(tokenized_lengths),
        "max_sequence_length": max(tokenized_lengths) if tokenized_lengths else 0,
        "total_tokens": len(all_tokens),
        "compression_ratio": total_chars / len(all_tokens) if all_tokens else 0,
        "unique_tokens": len(set(all_tokens)),
    }

    return stats
