# Loading model weights

The trainable model uses the same learned parameters and layer names as the
inference-only model. Adding a batch dimension and returning logits at every
token position changes the forward pass, but does not change parameter shapes.
The GPT-2 small translation described in
[`../llm_inference_only/WEIGHT_LOADING.md`](../llm_inference_only/WEIGHT_LOADING.md)
therefore still applies.

`weight_loading.py` supports two checkpoint types:

- `load_gpt2_small()` translates the external GPT-2 small checkpoint.
- `load_training_checkpoint()` loads the model configuration and state written
  by `pretrain.py`.

Both loaders put the model in evaluation mode before returning it.
