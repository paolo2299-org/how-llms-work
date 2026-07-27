"""Causal self-attention for a batch of token sequences."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, head_dim, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        out_dim = num_heads * head_dim

        if input_dim != out_dim:
            raise ValueError("input_dim must equal head_dim * num_heads")

        self.W_query = nn.Linear(input_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, out_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split each token vector into attention heads, then put the head axis
        # before the token axis: (batch, heads, tokens, head dimensions).
        queries = queries.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = keys.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = values.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

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
        context = context.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.num_heads * self.head_dim,
        )

        return self.out_proj(context)
