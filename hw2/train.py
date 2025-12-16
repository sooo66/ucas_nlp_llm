"""Training script for FNN n-gram, vanilla RNN, and Transformer decoder LMs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    compute_perplexity,
    count_parameters,
    evaluate_model,
    get_device,
    load_config,
    load_tokenizer,
    masked_cross_entropy,
    plot_step_curves,
    plot_training_curves,
    prepare_logger,
    read_corpus_lines,
    save_history,
    set_seed,
    split_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train language models on the Chinese news corpus.")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config TOML.")
    parser.add_argument("--model", choices=["fnn", "rnn", "transformer"], required=True, help="Model to train.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training (expects model+optimizer states).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run label for outputs/checkpoints (defaults to model name). Use the same value when resuming.",
    )
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


def build_datasets(
    model_name: str,
    cfg: ExperimentConfig,
    tokenizer,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Callable]:
    lines = read_corpus_lines(cfg.paths.dataset)
    train_lines, val_lines = split_corpus(lines, val_size=1000)
    train_sentences = tokenize_corpus(train_lines, tokenizer)
    val_sentences = tokenize_corpus(val_lines, tokenizer)
    if model_name == "fnn":
        train_dataset = FNNNgramDataset(train_sentences, cfg.fnn.context_size)
        val_dataset = FNNNgramDataset(val_sentences, cfg.fnn.context_size)
        collate_fn = fnn_collate
    else:
        seq_len = cfg.training.seq_len
        train_dataset = SequentialBlockDataset(train_sentences, seq_len)
        val_dataset = SequentialBlockDataset(val_sentences, seq_len)
        collate_fn = sequential_collate
    logger.info("Train samples: {} | Validation samples: {}", len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset, collate_fn


def _grad_norm_no_clip(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().pow(2).sum().item()
    return float(total ** 0.5) if total > 0 else 0.0


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pad_idx: int,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_interval: int,
    global_step: int,
) -> Tuple[float, float, float, int, list]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norm_total = 0.0
    grad_norm_steps = 0
    step_records = []
    interval = max(1, log_interval)
    progress = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
    for step, (inputs, targets) in enumerate(progress, 1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss, valid = masked_cross_entropy(logits, targets, pad_idx)
        loss.backward()
        global_step += 1
        if grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip).item()
        else:
            grad_norm = _grad_norm_no_clip(model)
        grad_norm_total += grad_norm
        grad_norm_steps += 1
        optimizer.step()
        total_loss += loss.item() * valid
        total_tokens += valid
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            ppl=f"{compute_perplexity(loss.item()):.2f}",
            grad=f"{grad_norm:.2f}",
        )
        if step % 100 == 0 or step == len(data_loader):
            logger.info(
                "Epoch {} Step {}/{} | loss={:.4f} ppl={:.2f}",
                epoch,
                step,
                len(data_loader),
                loss.item(),
                compute_perplexity(loss.item()),
            )
        if global_step % interval == 0 or step == len(data_loader):
            step_records.append({"step": global_step, "train_loss": loss.item(), "grad_norm": grad_norm})
    progress.close()
    mean_loss = total_loss / max(1, total_tokens)
    avg_grad_norm = grad_norm_total / max(1, grad_norm_steps)
    return mean_loss, compute_perplexity(mean_loss), avg_grad_norm, global_step, step_records


def run_training(
    model_name: str,
    config_path: str = "config.toml",
    reuse_logger: bool = False,
    cfg_override: Optional[ExperimentConfig] = None,
    run_name: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> Tuple[Path, Dict[str, List[float]]]:
    cfg = cfg_override or load_config(config_path)
    run_label = run_name or model_name
    output_root = Path(cfg.paths.output_dir)
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    if not reuse_logger:
        prepare_logger(cfg.logging.log_level, output_dir)
    set_seed(cfg.training.seed)
    tokenizer = load_tokenizer(cfg.paths.tokenizer_name)
    pad_idx = tokenizer.pad_token_id
    device = get_device(cfg.training.device)

    train_dataset, val_dataset, collate_fn = build_datasets(model_name, cfg, tokenizer)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    model = instantiate_model(model_name, cfg, vocab_size, pad_idx).to(device)
    logger.info("Training {} ({}) with {} parameters.", model_name, run_label, count_parameters(model))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    best_val_ppl = float("inf")
    best_ckpt_path = output_dir / f"{run_label}_best.pt"
    last_ckpt_path = output_dir / f"{run_label}_last.pt"
    global_step = 0
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "grad_norm": [],
        "step": [],
        "train_loss_step": [],
        "grad_norm_step": [],
        "val_step": [],
        "val_ppl_step": [],
    }

    start_epoch = 1
    if resume_path:
        ckpt_path = Path(resume_path)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get("model_state", state))
            if "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = state.get("epoch", 0) + 1
            global_step = state.get("global_step", 0)
            best_val_ppl = state.get("best_val_ppl", best_val_ppl)
            loaded_history = state.get("history")
            if isinstance(loaded_history, dict):
                history.update({k: loaded_history.get(k, v) for k, v in history.items()})
            logger.info("Resuming from {} (start at epoch {}, global_step {}).", ckpt_path, start_epoch, global_step)
        else:
            logger.warning("Resume checkpoint {} not found; starting fresh.", ckpt_path)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        train_loss, train_ppl, grad_norm, global_step, step_records = train_one_epoch(
            model,
            train_loader,
            optimizer,
            pad_idx,
            device,
            cfg.training.grad_clip,
            epoch,
            cfg.training.log_interval,
            global_step,
        )
        val_loss, val_ppl = evaluate_model(model, val_loader, pad_idx, device)
        logger.info(
            "Epoch {} summary | train_loss={:.4f} train_ppl={:.2f} | val_loss={:.4f} val_ppl={:.2f}",
            epoch,
            train_loss,
            train_ppl,
            val_loss,
            val_ppl,
        )
        for record in step_records:
            history["step"].append(record["step"])
            history["train_loss_step"].append(record["train_loss"])
            history["grad_norm_step"].append(record["grad_norm"])
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["grad_norm"].append(grad_norm)
        history["val_step"].append(global_step)
        history["val_ppl_step"].append(val_ppl)
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_ppl": best_val_ppl,
                    "history": history,
                },
                best_ckpt_path,
            )
            logger.info("New best checkpoint saved to {}", best_ckpt_path)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_ppl": best_val_ppl,
                "history": history,
            },
            last_ckpt_path,
        )
        logger.info("Latest checkpoint saved to {}", last_ckpt_path)

    logger.info("Best validation perplexity for {}: {:.2f}", run_label, best_val_ppl)
    plot_training_curves(history, output_dir, run_label)
    plot_step_curves(history, output_dir, run_label)
    save_history(history, output_root, run_label)
    return best_ckpt_path, history


def main() -> None:
    args = parse_args()
    run_training(args.model, args.config, run_name=args.run_name, resume_path=args.resume)


if __name__ == "__main__":
    main()
