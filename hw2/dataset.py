"""Dataset helpers for n-gram, RNN, and Transformer language models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from einops import rearrange
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


@dataclass
class TokenizedSentence:
    input_ids: List[int]


def tokenize_corpus(lines: Sequence[str], tokenizer: PreTrainedTokenizerFast) -> List[TokenizedSentence]:
    sentences: List[TokenizedSentence] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ids = tokenizer.encode(stripped, add_special_tokens=True)
        if len(ids) < 2:
            continue
        sentences.append(TokenizedSentence(input_ids=ids))
    if not sentences:
        raise ValueError("No tokens collected from the corpus.")
    logger.info("Tokenized {} sentences.", len(sentences))
    return sentences


class FNNNgramDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Generate (context, target) sliding windows for FNN model."""

    def __init__(self, sentences: Sequence[TokenizedSentence], context_size: int):
        if context_size < 1:
            raise ValueError("context_size must be >= 1.")
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for sent in sentences:
            ids = sent.input_ids
            if len(ids) <= context_size:
                continue
            for idx in range(context_size, len(ids)):
                context = torch.tensor(ids[idx - context_size : idx], dtype=torch.long)
                target = torch.tensor(ids[idx], dtype=torch.long)
                self.samples.append((context, target))
        if not self.samples:
            raise ValueError("Unable to build any n-gram samples.")
        logger.info("Prepared {} n-gram samples (context_size={}).", len(self.samples), context_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.samples[idx]


class SequentialBlockDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length blocks for RNN/Transformer teacher-forcing training."""

    def __init__(self, sentences: Sequence[TokenizedSentence], seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        flat_ids: List[int] = []
        for sent in sentences:
            flat_ids.extend(sent.input_ids)
        tokens = torch.tensor(flat_ids, dtype=torch.long)
        if tokens.numel() <= seq_len:
            raise ValueError("Not enough tokens for any training block.")
        self.tokens = tokens
        self.block_size = seq_len
        self.num_blocks = (tokens.numel() - 1) // seq_len
        logger.info(
            "Flattened {} tokens -> {} blocks (seq_len={} -> inputs len {}).",
            tokens.numel(),
            self.num_blocks,
            seq_len,
            seq_len - 1,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        start = idx * self.block_size
        end = start + self.block_size + 1
        block = self.tokens[start:end]
        inputs = rearrange(block[:-1], "t -> t")
        targets = rearrange(block[1:], "t -> t")
        return inputs.clone(), targets.clone()


def fnn_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    contexts = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return rearrange(contexts, "b n -> b n"), rearrange(targets, "b -> b")


def sequential_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return rearrange(inputs, "b l -> b l"), rearrange(targets, "b l -> b l")
