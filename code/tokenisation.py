from functools import lru_cache

import tiktoken


# GPT-2's weights only make sense with the token IDs they were trained with.
encoding = tiktoken.get_encoding("gpt2")


def tokenise(text):
    """Turn text into one unbatched sequence of token IDs."""
    return encoding.encode(text)


def detokenise(token_ids):
    """Turn one sequence of token IDs back into text."""
    return encoding.decode(token_ids)


def vocabulary_size():
    return encoding.n_vocab


@lru_cache
def valid_token_ids():
    """Return the token IDs that the encoding can decode."""
    valid_ids = []
    for token_id in range(encoding.n_vocab):
        try:
            encoding.decode_single_token_bytes(token_id)
        except KeyError:
            continue
        valid_ids.append(token_id)
    return tuple(valid_ids)
