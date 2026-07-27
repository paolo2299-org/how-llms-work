"""Generate one token with the batched model."""

import argparse
from pathlib import Path

import torch

from tokenisation import (
    detokenise,
    tokenise,
    valid_token_ids,
)
from weight_loading import (
    DEFAULT_WEIGHTS_PATH,
    load_gpt2_small,
    load_training_checkpoint,
)


def generate_next_token(model, prompt):
    prompt_token_ids = tokenise(prompt)
    device = next(model.parameters()).device

    # Inference uses a batch containing one prompt.
    token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        all_logits = model(token_ids)
        next_token_logits = all_logits[0, -1]
        decodable_token_ids = torch.tensor(
            valid_token_ids(),
            dtype=torch.long,
            device=device,
        )
        decodable_logits = next_token_logits[decodable_token_ids]
        next_token_probabilities = torch.softmax(decodable_logits, dim=-1)
        selected_index = torch.argmax(next_token_probabilities)
        next_token_id = decodable_token_ids[selected_index].item()

    return {
        "prompt_token_ids": prompt_token_ids,
        "next_token_id": next_token_id,
        "next_token": detokenise([next_token_id]),
        "probability": next_token_probabilities[selected_index].item(),
        "text": detokenise(prompt_token_ids + [next_token_id]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load model weights and generate one next token."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The dog fetched the",
        help="prompt to pass to the model",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="path to external gpt2-small.pth weights",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="use a checkpoint created by pretrain.py instead of GPT-2",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch inference device (default: cpu)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        model = load_training_checkpoint(args.checkpoint, args.device)
    else:
        model = load_gpt2_small(args.weights, args.device)

    result = generate_next_token(model, args.prompt)

    print(f"Prompt:         {args.prompt!r}")
    print(f"Prompt IDs:     {result['prompt_token_ids']}")
    print(f"Next token ID:  {result['next_token_id']}")
    print(f"Next token:     {result['next_token']!r}")
    print(f"Probability:    {result['probability']:.6f}")
    print(f"Combined text:  {result['text']!r}")


if __name__ == "__main__":
    main()
