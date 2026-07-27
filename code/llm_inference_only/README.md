# Run the inference-only LLM

## Install dependencies

From the repository root, use Python 3.12 to create a virtual environment and
install the requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/llm_inference_only/requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Generate output

Place the GPT-2 small checkpoint at `weights/gpt2-small.pth`, then run:

```bash
python code/llm_inference_only/generate.py "The dog fetched the"
```

The command prints the prompt's token IDs, the selected next token and its
probability, and the prompt with that token appended.

To use a checkpoint stored elsewhere:

```bash
python code/llm_inference_only/generate.py \
    --weights /path/to/gpt2-small.pth \
    "The dog fetched the"
```
