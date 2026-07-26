# Single-next-token LLM

This folder joins the site's code samples into one small, runnable language
model. It loads the GPT-2 small open weights, accepts one prompt (not a batch),
and generates exactly one next token.

The implementation keeps the tutorial's simple, unbatched structure while
matching the parameter-bearing parts of GPT-2 small. The loader translates the
checkpoint's parameter names, ignores its reproducible mask buffers, validates
every key and tensor shape, and then loads the weights strictly.

## Files

| File | Site section |
| --- | --- |
| `tokenisation.py` | Tokenisation and detokenisation with GPT-2's vocabulary |
| `token_embedding.py` | Token and positional embeddings |
| `self_attention.py` | Causal multi-head self-attention |
| `feed_forward.py` | Feed-forward layer |
| `transformer.py` | LayerNorm, residual connections, and a transformer block |
| `language_model.py` | Transformer stack and vocabulary projection |
| `weight_loading.py` | GPT-2 small construction, validation, and weight loading |
| `generate.py` | Weight loading and greedy selection of one next token |

The tensors deliberately have no batch dimension. Their main shapes are:

```text
token IDs                 (number of tokens)
token vectors             (number of tokens, model dimension)
next-token logits         (vocabulary size)
```

The compatibility-specific changes are limited to GPT-2's tokenizer and
vocabulary, configurable query/key/value biases, biases on attention output
projections, a bias-free vocabulary projection, and GPT-2's GELU approximation.
Batching and dropout are not required to use trained weights for inference.

## Install

Use Python 3.12, matching the rest of this repository. From the repository
root, create an isolated environment and install the two dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Weights

The example expects the GPT-2 small checkpoint at:

```text
weights/gpt2-small.pth
```

The path is resolved relative to this repository, so the script works from any
current directory. The weights file is kept out of Git because of its size.
Pass `--weights` to use the same checkpoint from a different location.

## Run

Pass a prompt as the sole argument:

```bash
python code/generate.py "The dog fetched the"
```

To supply a different path:

```bash
python code/generate.py \
  --weights /path/to/gpt2-small.pth \
  "The dog fetched the"
```

Or run without arguments to use the default path and example prompt:

```bash
python code/generate.py
```

The script prints the prompt's token IDs, the selected next-token ID and text,
its probability, and the prompt with that one token appended.

Prompts must contain at least one token and fit within GPT-2 small's
1,024-token context length.
