# Loading GPT-2 small weights

`weight_loading.py` constructs the fixed GPT-2 small architecture, translates
the checkpoint's parameter names to the teaching model's descriptive names,
validates every key and tensor shape, and loads the result strictly.

The expected checkpoint is `weights/gpt2-small.pth`. Causal attention masks are
not learned values, so the loader discards the checkpoint copies and the model
recreates a mask for the current sequence during each forward pass.

This loader intentionally supports only that checkpoint and architecture. The
small model created by `pretrain.py` has different dimensions and is saved only
to demonstrate the training process.
