"""Evaluate checkpoints and visualize metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import DataLoader

from dataset import (
    FNNNgramDataset,
    SequentialBlockDataset,
    fnn_collate,
    sequential_collate,
    tokenize_corpus,
)
from models import FNNNgramLM, TransformerDecoderLM, VanillaRNNLM
from utils import (
    ExperimentConfig,
    evaluate_model,
    get_device,
    load_config,
    load_tokenizer,
    plot_eval_metrics,
    prepare_logger,
    read_corpus_lines,
    set_seed,
    split_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints.")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config.")
    parser.add_argument("--model", choices=["fnn", "rnn", "transformer"], required=True, help="Model to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path from training.")
    return parser.parse_args()


def instantiate_model(model_name: str, cfg: ExperimentConfig, vocab_size: int, pad_idx: int) -> torch.nn.Module:
    if model_name == "fnn":
        return FNNNgramLM(
            vocab_size=vocab_size,
            context_size=cfg.fnn.context_size,
            embedding_dim=cfg.fnn.embedding_dim,
            hidden_dim=cfg.fnn.hidden_dim,
            pad_idx=pad_idx,
        )
    if model_name == "rnn":
        return VanillaRNNLM(
            vocab_size=vocab_size,
            embedding_dim=cfg.rnn.embedding_dim,
            hidden_dim=cfg.rnn.hidden_dim,
            pad_idx=pad_idx,
        )
    return TransformerDecoderLM(
        vocab_size=vocab_size,
        d_model=cfg.transformer.d_model,
        num_layers=cfg.transformer.num_layers,
        num_heads=cfg.transformer.num_heads,
        ffn_dim=cfg.transformer.ffn_dim,
        dropout=cfg.transformer.dropout,
        max_seq_len=cfg.training.seq_len,
        pad_idx=pad_idx,
    )


def build_validation_loader(
    model_name: str,
    cfg: ExperimentConfig,
    tokenizer,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    lines = read_corpus_lines(cfg.paths.dataset)
    _, val_lines = split_corpus(lines, val_size=1000)
    val_sentences = tokenize_corpus(val_lines, tokenizer)
    if model_name == "fnn":
        dataset = FNNNgramDataset(val_sentences, cfg.fnn.context_size)
        collate_fn: Callable = fnn_collate
    else:
        dataset = SequentialBlockDataset(val_sentences, cfg.training.seq_len)
        collate_fn = sequential_collate
    logger.info("Validation samples: {}", len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def run_evaluation(
    model_name: str,
    checkpoint_path: str,
    config_path: str = "config.toml",
    reuse_logger: bool = False,
    cfg_override: Optional[ExperimentConfig] = None,
    run_name: Optional[str] = None,
) -> Tuple[float, float]:
    cfg = cfg_override or load_config(config_path)
    run_label = run_name or model_name
    run_dir = Path(cfg.paths.output_dir) / run_label
    eval_dir = run_dir / "eval"
    if not reuse_logger:
        prepare_logger(cfg.logging.log_level, eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.training.seed)
    tokenizer = load_tokenizer(cfg.paths.tokenizer_name)
    pad_idx = tokenizer.pad_token_id
    device = get_device(cfg.training.device)

    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = instantiate_model(model_name, cfg, vocab_size, pad_idx).to(device)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found.")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    logger.info("Loaded checkpoint {}.", checkpoint)

    val_loader = build_validation_loader(
        model_name,
        cfg,
        tokenizer,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loss, val_ppl = evaluate_model(model, val_loader, pad_idx, device)
    logger.info("[{}] Validation loss={:.4f} perplexity={:.2f}", run_label, val_loss, val_ppl)
    plot_eval_metrics({"loss": val_loss, "perplexity": val_ppl}, run_dir, run_label)
    return val_loss, val_ppl


def main() -> None:
    args = parse_args()
    run_evaluation(args.model, args.checkpoint, args.config)


if __name__ == "__main__":
    main()
