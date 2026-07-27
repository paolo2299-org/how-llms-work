# Run and pre-train the batched LLM

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

## Pre-train the small teaching model

The included `sample.txt` file is intentionally tiny. It demonstrates the
training process, but cannot produce a useful language model.

```bash
python code/llm/pretrain.py
```

The command tokenises the text, forms batches of input/target sequences, trains
for one epoch, and saves a checkpoint to
`weights/tiny-teaching-model.pth`.

The main options can be changed directly:

```bash
python code/llm/pretrain.py \
    --input code/llm/sample.txt \
    --epochs 2 \
    --batch-size 4 \
    --sequence-length 16
```

## Generate from the small checkpoint

```bash
python code/llm/generate.py \
    --checkpoint weights/tiny-teaching-model.pth \
    "The lantern"
```

The output will usually be poor because the model and input are deliberately
small.

## Generate with GPT-2 small

Place the GPT-2 small checkpoint at `weights/gpt2-small.pth`, then run:

```bash
python code/llm/generate.py "The dog fetched the"
```

To use GPT-2 weights stored elsewhere:

```bash
python code/llm/generate.py \
    --weights /path/to/gpt2-small.pth \
    "The dog fetched the"
```

See `ADDITIONS.md` for a concise comparison with the inference-only version.
