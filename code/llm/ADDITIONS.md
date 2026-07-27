# Additions for pre-training

Compared with `../llm_inference_only`, this version adds only the pieces needed
to train on next-token prediction:

- Inputs include a batch dimension: `(batch, tokens)`.
- Attention keeps that batch dimension while splitting and recombining heads.
- The model returns vocabulary logits for every position:
  `(batch, tokens, vocabulary)`.
- `NextTokenDataset` makes input/target pairs by shifting each token sequence
  one position.
- `DataLoader` groups those pairs into shuffled batches.
- `pretrain.py` applies cross-entropy loss, backpropagation, and AdamW updates,
  then saves the model configuration and weights.
- `sample.txt` provides a tiny illustrative training corpus.
- `generate.py` can load either external GPT-2 weights or a checkpoint produced
  by `pretrain.py`.

The transformer layers and their learned parameter shapes are unchanged.
