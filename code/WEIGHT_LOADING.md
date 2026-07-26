# Loading the GPT-2 small weights

This document records the implementation decisions for loading one specific
checkpoint into the simple language model in this directory:

```text
weights/gpt2-small.pth
```

No other checkpoint format or model variant is in scope. Do not add presets,
metadata discovery, fine-tuned-model support, classifier support, or a generic
checkpoint abstraction.

The goal is GPT-2 small inference while keeping the teaching model recognisable
from the HTML tutorial samples. Do not add batching, dropout,
checkpoint-compatible attribute names, or stored causal masks merely to make
loading easier.

## Current compatibility baseline

The model already contains the small changes that affect GPT-2's learned
parameters or inference calculations:

- `tokenisation.py` uses the GPT-2 tokenizer and its 50,257-token vocabulary.
- Query, key, and value biases are configurable through `qkv_bias`.
- The attention output projection has a bias.
- The final vocabulary projection does not have a bias.
- The feed-forward layer uses `nn.GELU(approximate="tanh")`, matching GPT-2's
  GELU approximation.

The remaining differences are intentional:

- The teaching model processes one unbatched token sequence.
- It has no dropout modules.
- It creates the causal attention mask during each forward pass.
- It calculates logits only for the final input token.
- Its Python attribute names favour the tutorial's terminology rather than the
  checkpoint's abbreviated names.

None of these differences prevents correct inference with the GPT-2 weights.

## Construct the fixed GPT-2 small shape

Always construct the model with this configuration:

```python
model = LanguageModel(
    vocab_size=50257,
    max_sequence_length=1024,
    model_dim=768,
    head_dim=64,
    num_heads=12,
    hidden_dim=3072,
    num_layers=12,
    qkv_bias=True,
)
```

Do not use the tiny random model configuration from `generate.py`; its tensor
shapes cannot accept GPT-2 small weights.

## Read the checkpoint safely

Read the state dictionary on the CPU using PyTorch's weights-only mode:

```python
source_state = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)
```

The loader defaults `weights_path` to the repository-local
`weights/gpt2-small.pth`, but it should still accept an explicit path.

After translating and validating the state dictionary, load it strictly, call
`model.eval()`, and then move the model to the requested inference device.

This teaching implementation does not need streaming, sharded loading, or
automatic multi-device placement.

## Drop the causal-mask buffers

The checkpoint contains one entry per transformer block:

```text
trf_blocks.0.att.mask
trf_blocks.1.att.mask
...
trf_blocks.11.att.mask
```

These are registered buffers, not trained parameters. They are triangular
matrices that prevent attention to future tokens.

The teaching model reconstructs the same mask from the current sequence length
inside `MultiHeadAttention.forward`. Drop every source key ending in
`.att.mask`; do not add stored masks to the teaching model.

Ignoring these entries loses no learned information.

## Translate checkpoint attribute names

PyTorch matches `state_dict` entries by their exact string keys. The checkpoint
and teaching model use different names for equivalent layers, so translate the
keys before calling `load_state_dict`.

### Top-level parameters

| GPT-2 checkpoint key | Teaching-model key |
| --- | --- |
| `tok_emb.weight` | `embedding.token_embedding.weight` |
| `pos_emb.weight` | `embedding.position_embedding.weight` |
| `final_norm.scale` | `final_norm.weight` |
| `final_norm.shift` | `final_norm.bias` |
| `out_head.weight` | `vocabulary_projection.weight` |

### Parameters repeated for every transformer block

For each block number `{i}` from `0` through `11`:

| GPT-2 checkpoint key | Teaching-model key |
| --- | --- |
| `trf_blocks.{i}.att.W_query.weight` | `transformer_blocks.{i}.attention.W_query.weight` |
| `trf_blocks.{i}.att.W_query.bias` | `transformer_blocks.{i}.attention.W_query.bias` |
| `trf_blocks.{i}.att.W_key.weight` | `transformer_blocks.{i}.attention.W_key.weight` |
| `trf_blocks.{i}.att.W_key.bias` | `transformer_blocks.{i}.attention.W_key.bias` |
| `trf_blocks.{i}.att.W_value.weight` | `transformer_blocks.{i}.attention.W_value.weight` |
| `trf_blocks.{i}.att.W_value.bias` | `transformer_blocks.{i}.attention.W_value.bias` |
| `trf_blocks.{i}.att.out_proj.weight` | `transformer_blocks.{i}.attention.out_proj.weight` |
| `trf_blocks.{i}.att.out_proj.bias` | `transformer_blocks.{i}.attention.out_proj.bias` |
| `trf_blocks.{i}.ff.layers.0.weight` | `transformer_blocks.{i}.feed_forward.expand.weight` |
| `trf_blocks.{i}.ff.layers.0.bias` | `transformer_blocks.{i}.feed_forward.expand.bias` |
| `trf_blocks.{i}.ff.layers.2.weight` | `transformer_blocks.{i}.feed_forward.project.weight` |
| `trf_blocks.{i}.ff.layers.2.bias` | `transformer_blocks.{i}.feed_forward.project.bias` |
| `trf_blocks.{i}.norm1.scale` | `transformer_blocks.{i}.attention_norm.weight` |
| `trf_blocks.{i}.norm1.shift` | `transformer_blocks.{i}.attention_norm.bias` |
| `trf_blocks.{i}.norm2.scale` | `transformer_blocks.{i}.feed_forward_norm.weight` |
| `trf_blocks.{i}.norm2.shift` | `transformer_blocks.{i}.feed_forward_norm.bias` |

After dropping mask keys and translating names, the dictionary should match the
model exactly. Do not hide mistakes with `strict=False`:

```python
model.load_state_dict(translated_state, strict=True)
```

## Why batching is unnecessary

Batching changes only the leading dimensions used during the forward pass. It
does not add or alter any learned parameters.

The source GPT model receives `(batch, tokens, dimensions)`. The teaching model
receives `(tokens, dimensions)`. For a single prompt, both apply the same
learned projections and transformer calculations to the same sequence.

## Why dropout is unnecessary

Dropout has no trained parameters. GPT-2 small uses a zero dropout rate in the
source configuration, and inference also runs under `model.eval()`.

Adding dropout modules would therefore add tutorial complexity without changing
GPT-2 small inference.

## Why the dynamic mask is valid

The mask is a deterministic function of sequence length. Whether it is stored
once in the model or reconstructed during the forward pass does not change the
attention calculation.

The loader should discard the checkpoint copies rather than changing the
teaching model's existing mask logic.

## Why final-token-only output is valid

The source GPT model calculates vocabulary logits for every input position. The
teaching model applies the same final normalisation and vocabulary projection
only to the final token vector.

For next-token generation, this is equivalent to selecting:

```python
all_logits[0, -1]
```

from the source model. Earlier-position logits are not required.

## Validation requirements

The loader should:

1. Construct the fixed GPT-2 small model configuration above.
2. Read the checkpoint with `map_location="cpu"` and `weights_only=True`.
3. Drop only keys ending in `.att.mask`.
4. Translate every remaining source key.
5. Reject unknown source keys.
6. Assert that translated keys exactly equal `model.state_dict().keys()`.
7. Assert that every translated tensor shape matches its destination shape.
8. Load with `strict=True`.
9. Call `model.eval()`.
10. Run a fixed-prompt smoke test.
11. Compare next-token logits with a known-good GPT-2 implementation when
    changing the loader.

This architecture has already been checked against `gpt2-small.pth`: all 197
learned entries map with no missing, extra, or shape-mismatched entries after
the 12 mask buffers are excluded.

## Keep loading outside the model classes

Put checkpoint-specific work in a small loader module. It should own:

- constructing the fixed GPT-2 small `LanguageModel`;
- reading `gpt2-small.pth`;
- dropping mask buffers;
- translating names;
- validating keys and shapes;
- loading the translated dictionary;
- selecting the inference device.

Keep the model files focused on explaining the LLM. In particular, do not
rename the tutorial's attributes merely to mirror this one external checkpoint.
