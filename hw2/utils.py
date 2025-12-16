"""Utility helpers for configuration, logging, and evaluation."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import toml
from einops import reduce, rearrange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@dataclass
class PathsConfig:
    dataset: str
    tokenizer_name: str
    output_dir: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    seq_len: int
    seed: int
    device: str


@dataclass
class FNNConfig:
    context_size: int
    embedding_dim: int
    hidden_dim: int


@dataclass
class RNNConfig:
    embedding_dim: int
    hidden_dim: int


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float


@dataclass
class LoggingConfig:
    log_level: str


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    training: TrainingConfig
    fnn: FNNConfig
    rnn: RNNConfig
    transformer: TransformerConfig
    logging: LoggingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    cfg_dict = toml.load(path)
    return ExperimentConfig(
        paths=PathsConfig(**cfg_dict["paths"]),
        training=TrainingConfig(**cfg_dict["training"]),
        fnn=FNNConfig(**cfg_dict["fnn"]),
        rnn=RNNConfig(**cfg_dict["rnn"]),
        transformer=TransformerConfig(**cfg_dict["transformer"]),
        logging=LoggingConfig(**cfg_dict["logging"]),
    )


def prepare_logger(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=level.upper())
    logger.add(output_dir / "run.log", level=level.upper(), rotation="5 MB", enqueue=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Tokenizer must be a fast tokenizer implementation.")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def get_device(name: str) -> torch.device:
    desired = name.lower()
    if desired == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        desired = "cpu"
    return torch.device(desired)


def read_corpus_lines(file_path: str | Path) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file {path} not found.")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1000:
        raise ValueError("Corpus must contain more than 1000 sentences.")
    return lines


def split_corpus(lines: List[str], val_size: int = 1000) -> Tuple[List[str], List[str]]:
    if val_size >= len(lines):
        raise ValueError("Validation size must be smaller than the corpus.")
    return lines[:-val_size], lines[-val_size:]


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, pad_idx: int
) -> Tuple[torch.Tensor, int]:
    if logits.dim() == 3:
        logits_flat = rearrange(logits, "b t v -> (b t) v")
        targets_flat = rearrange(targets, "b t -> (b t)")
    else:
        logits_flat = logits
        targets_flat = targets
    mask = targets_flat != pad_idx
    valid = int(reduce(mask.int(), "n ->", "sum").item())
    if valid == 0:
        raise ValueError("No valid tokens present for loss computation.")
    loss = F.cross_entropy(logits_flat[mask], targets_flat[mask], reduction="mean")
    return loss, valid


def compute_perplexity(loss: float) -> float:
    return float(torch.exp(torch.tensor(loss)))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _ensure_fig_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path, model_name: str) -> None:
    if not history["epoch"]:
        return
    fig_dir = output_dir / "figures"
    _ensure_fig_dir(fig_dir)
    epochs = history["epoch"]

    def _plot_lines(data: Dict[str, List[float]], title: str, y_label: str, filename: str) -> None:
        plt.figure()
        for label, values in data.items():
            plt.plot(epochs, values, marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{model_name}_{filename}.png")
        plt.close()

    _plot_lines(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        "Train vs Val Loss",
        "Loss",
        "loss_curve",
    )
    _plot_lines(
        {"train_ppl": history["train_ppl"], "val_ppl": history["val_ppl"]},
        "Train vs Val Perplexity",
        "Perplexity",
        "ppl_curve",
    )
    if any(history["grad_norm"]):
        _plot_lines(
            {"grad_norm": history["grad_norm"]},
            "Average Gradient Norm",
            "L2 Norm",
            "grad_norm",
        )


def plot_eval_metrics(metrics: Dict[str, float], output_dir: Path, model_name: str) -> None:
    fig_dir = output_dir / "figures"
    _ensure_fig_dir(fig_dir)
    names = list(metrics.keys())
    values = [metrics[name] for name in names]
    plt.figure()
    bars = plt.bar(names, values, color=["#2b8cbe", "#e34a33", "#31a354", "#756bb1"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    plt.title(f"{model_name} evaluation metrics")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_eval_metrics.png")
    plt.close()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss, valid = masked_cross_entropy(logits, targets, pad_idx)
        total_loss += loss.item() * valid
        total_tokens += valid
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, compute_perplexity(mean_loss)
