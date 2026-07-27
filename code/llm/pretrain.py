"""Pre-train a very small language model on a text file."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from language_model import LanguageModel
from tokenisation import tokenise, vocabulary_size


DEFAULT_TEXT_PATH = Path(__file__).with_name("sample.txt")
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "weights"
    / "tiny-teaching-model.pth"
)


class NextTokenDataset(Dataset):
    """Split text into fixed-length input and target sequences."""

    def __init__(self, text, sequence_length):
        self.token_ids = torch.tensor(tokenise(text), dtype=torch.long)
        self.sequence_length = sequence_length

        # Each target is the corresponding input shifted one token to the left.
        # Non-overlapping starts keep this example small and easy to inspect.
        self.start_positions = list(
            range(
                0,
                len(self.token_ids) - sequence_length,
                sequence_length,
            )
        )

        if not self.start_positions:
            raise ValueError(
                "the input text must contain more tokens than sequence_length"
            )

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, index):
        start = self.start_positions[index]
        stop = start + self.sequence_length
        inputs = self.token_ids[start:stop]
        targets = self.token_ids[start + 1 : stop + 1]
        return inputs, targets


def create_data_loader(text_path, sequence_length, batch_size):
    text = Path(text_path).read_text(encoding="utf-8")
    dataset = NextTokenDataset(text, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, batches, epochs, learning_rate, device):
    model.to(device)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, targets in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimiser.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
            )
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        average_loss = total_loss / len(batches)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train a tiny teaching model on a text file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="plain-text training input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="where to save the trained checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(1)

    model_config = {
        "vocab_size": vocabulary_size(),
        "max_sequence_length": args.sequence_length,
        "model_dim": 32,
        "head_dim": 8,
        "num_heads": 4,
        "hidden_dim": 128,
        "num_layers": 2,
    }
    model = LanguageModel(**model_config)
    batches = create_data_loader(
        args.input,
        args.sequence_length,
        args.batch_size,
    )

    print(
        f"Training on {len(batches.dataset)} sequences "
        f"in {len(batches)} batches."
    )
    train(
        model,
        batches,
        args.epochs,
        args.learning_rate,
        args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "model_state": model.state_dict(),
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
