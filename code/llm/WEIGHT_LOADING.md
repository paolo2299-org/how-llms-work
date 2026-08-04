# Loading model weights

`generate.py` can load GPT-2 small weights or a checkpoint created by
`pretrain.py`.

## GPT-2 small weights

Save `gpt2-small.pth` in the repository's `weights` directory, then run:

```bash
python code/llm/generate.py "The dog fetched the"
```

To use a file stored elsewhere:

```bash
python code/llm/generate.py \
    --weights /path/to/gpt2-small.pth \
    "The dog fetched the"
```

## Checkpoints created by `pretrain.py`

Create the default teaching-model checkpoint:

```bash
python code/llm/pretrain.py
```

Load it for generation:

```bash
python code/llm/generate.py \
    --checkpoint weights/tiny-teaching-model.pth \
    "The dog"
```

Use `--device cuda` or another PyTorch device to run either model somewhere
other than the CPU.
