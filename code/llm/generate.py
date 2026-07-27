import argparse
from pathlib import Path

import torch

from tokenisation import (
    detokenise,
    tokenise,
    valid_token_ids,
)
from weight_loading import DEFAULT_WEIGHTS_PATH, load_gpt2_small


def generate_next_token(model, prompt):
    prompt_token_ids = tokenise(prompt)
    device = next(model.parameters()).device
    token_ids = torch.tensor(
        prompt_token_ids,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        next_token_logits = model(token_ids)[-1]
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
        description="Load GPT-2 small and generate one next token."
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
        help="path to gpt2-small.pth",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch inference device (default: cpu)",
    )
    args = parser.parse_args()

    model = load_gpt2_small(args.weights, device=args.device)
    result = generate_next_token(model, args.prompt)

    print(f"Prompt:         {args.prompt!r}")
    print(f"Prompt IDs:     {result['prompt_token_ids']}")
    print(f"Next token ID:  {result['next_token_id']}")
    print(f"Next token:     {result['next_token']!r}")
    print(f"Probability:    {result['probability']:.6f}")
    print(f"Combined text:  {result['text']!r}")


if __name__ == "__main__":
    main()
