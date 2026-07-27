# Additions for pre-training

The `llm` version adds only the pieces needed to illustrate pre-training:

- `LanguageModel.forward` returns vocabulary logits for every input token,
  rather than only the final token. The inference script selects the final row.
- `pretrain.py` turns tokenised text into input/next-token training pairs,
  calculates cross-entropy loss, updates model parameters with AdamW, and saves
  the resulting state dictionary.
- `pretraining_input.txt` provides a tiny, original example corpus.
- `README.md` explains how to run the training example and its limitations.

The architecture remains unbatched and intentionally straightforward. The
small training configuration is separate from the GPT-2 small configuration;
its output is illustrative and is not expected to generate useful text.
