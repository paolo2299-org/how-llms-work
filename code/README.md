# Single-next-token LLM

This folder joins the site's code samples into one small, runnable language
model. It accepts one prompt (not a batch) and generates exactly one next token.

The model is intentionally **not trained** and cannot load pretrained weights.
Every model weight is initialised randomly. Its generated token is therefore
arbitrary, not a useful prediction. The example is for following the complete
path from text to a next-token prediction without adding training, weight
loading, or batching code.

## Files

| File | Site section |
| --- | --- |
| `tokenisation.py` | Tokenisation and detokenisation with `cl100k_base` |
| `token_embedding.py` | Token and positional embeddings |
| `self_attention.py` | Causal multi-head self-attention |
| `feed_forward.py` | Feed-forward layer |
| `transformer.py` | LayerNorm, residual connections, and a transformer block |
| `language_model.py` | Transformer stack and vocabulary projection |
| `generate.py` | Greedy selection of one next token |

The tensors deliberately have no batch dimension. Their main shapes are:

```text
token IDs                 (number of tokens)
token vectors             (number of tokens, model dimension)
next-token logits         (vocabulary size)
```

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

## Run

Pass a prompt as the sole argument:

```bash
python code/generate.py "The dog fetched the"
```

Or run without an argument to use that example prompt:

```bash
python code/generate.py
```

The script prints the prompt's token IDs, the selected next-token ID and text,
its probability, and the prompt with that one token appended.

Prompts must contain at least one token and fit within the example's
64-token context length. The fixed random seed makes repeated runs
reproducible with the same dependency versions.
