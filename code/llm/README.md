# Pre-train and run the teaching LLM

This version extends the inference-only model with the smallest practical
training example. It returns next-token predictions at every input position,
so `pretrain.py` can compare each prediction with the following token.

## Install dependencies

From the repository root, use Python 3.12 to create a virtual environment and
install the requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/llm/requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Pre-train the small model

The included `pretraining_input.txt` is intentionally tiny. It demonstrates
the mechanics of training, but it cannot produce a useful language model.

```bash
python code/llm/pretrain.py
```

The script tokenises the text, creates input/next-token pairs, trains a small
randomly initialised model, prints the loss, and saves its state dictionary as
`code/llm/pretrained_model.pth`. Try `--epochs 1` for a quicker run, or use
`--help` to see the input, output, sequence length, learning rate, and device
options.

## Generate with GPT-2 weights

The original GPT-2 inference example remains available. Place the GPT-2 small
checkpoint at `weights/gpt2-small.pth`, then run:

```bash
python code/llm/generate.py "The dog fetched the"
```

The generated teaching checkpoint uses a much smaller architecture, so it is
not interchangeable with the GPT-2 small checkpoint expected by `generate.py`.
