# GPT-2 small weights

The examples use Sebastian Raschka's public PyTorch conversion of the original
OpenAI GPT-2 small (124M) weights. Download
[`gpt2-small-124M.pth`](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth?download=true)
and save it under this repository-local name:

```text
weights/gpt2-small.pth
```

From the repository root, `curl` can download and rename it in one step:

```bash
curl --location --fail \
    --output weights/gpt2-small.pth \
    'https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth'
```

Raschka's
[`LLMs-from-scratch` alternative weight-loading notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/02_alternative_weight_loading/weight-loading-pytorch.ipynb)
documents the same checkpoint and source. The file is 702,538,513 bytes
(approximately 670 MiB); its SHA-256 digest is
`24a9078c5b27137fb2706d2206349c759952a7de8f72ef8e305dc02511bcabf8`.
It is intentionally ignored by Git.
