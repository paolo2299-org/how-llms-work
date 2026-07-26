import argparse

import torch

from language_model import LanguageModel
from tokenisation import (
    detokenise,
    tokenise,
    valid_token_ids,
    vocabulary_size,
)


MAX_SEQUENCE_LENGTH = 64


def build_model():
    # A fixed seed makes the randomly initialised teaching model reproducible.
    torch.manual_seed(42)
    return LanguageModel(
        vocab_size=vocabulary_size(),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        model_dim=32,
        head_dim=8,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
    )


def generate_next_token(model, prompt):
    prompt_token_ids = tokenise(prompt)
    token_ids = torch.tensor(prompt_token_ids, dtype=torch.long)

    model.eval()
    with torch.inference_mode():
        next_token_logits = model(token_ids)
        decodable_token_ids = torch.tensor(valid_token_ids(), dtype=torch.long)
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
        description="Run an untrained teaching LLM to generate one next token."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The dog fetched the",
        help="prompt to pass to the model",
    )
    args = parser.parse_args()

    result = generate_next_token(build_model(), args.prompt)

    print(f"Prompt:         {args.prompt!r}")
    print(f"Prompt IDs:     {result['prompt_token_ids']}")
    print(f"Next token ID:  {result['next_token_id']}")
    print(f"Next token:     {result['next_token']!r}")
    print(f"Probability:    {result['probability']:.6f}")
    print(f"Combined text:  {result['text']!r}")


if __name__ == "__main__":
    main()
