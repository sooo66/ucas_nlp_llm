"""Language model implementations leveraging einops for tensor ops."""
from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import nn


class FNNNgramLM(nn.Module):
    """Embedding lookup -> concat -> tanh -> softmax."""

    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(context_tokens)  # (batch, context, embed)
        flat_context = rearrange(embeddings, "b n d -> b (n d)")
        hidden = torch.tanh(self.hidden(flat_context))
        return self.output(hidden)


class VanillaRNNCell(nn.Module):
    """Manual vanilla RNN cell relying on einsum for projections."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Wxh = nn.Parameter(torch.randn(input_dim, hidden_dim) * (1 / math.sqrt(input_dim)))
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * (1 / math.sqrt(hidden_dim)))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        input_term = einsum(x_t, self.Wxh, "b d, d h -> b h")
        state_term = einsum(h_prev, self.Whh, "b h, h k -> b k")
        return torch.tanh(input_term + state_term + self.bias)


class VanillaRNNLM(nn.Module):
    """Autoregressive LM built from the custom vanilla RNN cell."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn_cell = VanillaRNNCell(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_tokens.size()
        embeddings = self.embedding(input_tokens)
        h_t = embeddings.new_zeros((batch_size, self.hidden_dim))
        logits = []
        for t in range(seq_len):
            h_t = self.rnn_cell(embeddings[:, t, :], h_t)
            logits.append(self.output(h_t).unsqueeze(1))
        return torch.cat(logits, dim=1)


class MultiHeadCausalSelfAttention(nn.Module):
    """Scaled dot-product attention expressed with einsum operations."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)
        attn_scores = einsum(q, k, "b h t d, b h s d -> b h t s") * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = einsum(attn_weights, v, "b h t s, b h s d -> b h t d")
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")
        return self.proj_dropout(self.out_proj(attn_output))


class PositionwiseFFN(nn.Module):
    """Two-layer FFN with ReLU non-linearity."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderLayer(nn.Module):
    """GPT-style decoder layer with causal attention."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadCausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, ffn_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransformerDecoderLM(nn.Module):
    """Decoder-only Transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int,
        pad_idx: int,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_tokens.size()
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds model positional capacity.")
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_tokens) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
