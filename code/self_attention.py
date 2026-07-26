import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, head_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        out_dim = num_heads * head_dim

        if input_dim != out_dim:
            raise ValueError("input_dim must equal head_dim * num_heads")

        self.W_query = nn.Linear(input_dim, out_dim, bias=False)
        self.W_key = nn.Linear(input_dim, out_dim, bias=False)
        self.W_value = nn.Linear(input_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        num_tokens = x.shape[0]

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split each token's vector into (num_heads, head_dim), then
        # move the head axis to the front so each head is processed in parallel.
        queries = queries.view(
            num_tokens, self.num_heads, self.head_dim
        ).transpose(0, 1)
        keys = keys.view(
            num_tokens, self.num_heads, self.head_dim
        ).transpose(0, 1)
        values = values.view(
            num_tokens, self.num_heads, self.head_dim
        ).transpose(0, 1)

        attention_scores = queries @ keys.transpose(-2, -1)

        # A token can only attend to itself and earlier tokens.
        mask = torch.triu(
            torch.ones(
                num_tokens,
                num_tokens,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(
            attention_scores / self.head_dim**0.5,
            dim=-1,
        )

        context = attention_weights @ values
        context = context.transpose(0, 1).reshape(
            num_tokens,
            self.num_heads * self.head_dim,
        )

        return self.out_proj(context)
