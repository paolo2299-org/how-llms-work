# Teaching language models

This directory contains two deliberately simple versions of the model:

- [`llm_inference_only`](llm_inference_only/) contains the original model. It
  loads pre-trained GPT-2 small weights and generates a next token.
- [`llm`](llm/) extends that model with the minimal code needed to pre-train a
  small model from text.

See each subdirectory's README for setup and usage instructions. See
[`ADDITIONS.md`](ADDITIONS.md) for a concise comparison of the two versions.
