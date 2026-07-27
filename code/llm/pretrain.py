import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from language_model import LanguageModel
from tokenisation import tokenise, vocabulary_size


DEFAULT_INPUT_PATH = Path(__file__).with_name("pretraining_input.txt")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("pretrained_model.pth")


def make_training_examples(token_ids, sequence_length):
    """Split text into sequences and the next tokens they should predict."""
    examples = []
    for start in range(0, len(token_ids) - 1, sequence_length):
        end = min(start + sequence_length, len(token_ids) - 1)
        inputs = token_ids[start:end]
        targets = token_ids[start + 1 : end + 1]
        examples.append((inputs, targets))
    return examples


def build_small_model(sequence_length):
    """Create a deliberately small model that can be trained on a laptop."""
    return LanguageModel(
        vocab_size=vocabulary_size(),
        max_sequence_length=sequence_length,
        model_dim=64,
        head_dim=16,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
    )


def pretrain(model, examples, epochs, learning_rate, device):
    model.to(device)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, target_ids in examples:
            inputs = torch.tensor(input_ids, dtype=torch.long, device=device)
            targets = torch.tensor(target_ids, dtype=torch.long, device=device)

            optimiser.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        average_loss = total_loss / len(examples)
        print(f"Epoch {epoch + 1:>2}/{epochs}: loss = {average_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pre-train the small teaching LLM.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    examples = make_training_examples(tokenise(text), args.sequence_length)
    if not examples:
        raise ValueError("input must contain at least two tokens")

    model = build_small_model(args.sequence_length)
    pretrain(model, examples, args.epochs, args.learning_rate, args.device)
    torch.save(model.state_dict(), args.output)
    print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
