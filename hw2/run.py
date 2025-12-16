"""Pipeline entry point to train and evaluate all language models."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from evaluate import run_evaluation
from train import run_training
from utils import load_config, prepare_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all language model variants.")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config TOML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg.paths.output_dir)
    prepare_logger(cfg.logging.log_level, output_dir)

    models = ["fnn", "rnn", "transformer"]
    for model_name in models:
        logger.info("========== Training {} ==========", model_name.upper())
        checkpoint_path = run_training(model_name, args.config, reuse_logger=True)
        logger.info("========== Evaluating {} ==========", model_name.upper())
        if checkpoint_path.exists():
            run_evaluation(model_name, str(checkpoint_path), args.config, reuse_logger=True)
        else:
            logger.warning("Checkpoint {} missing; skipping evaluation.", checkpoint_path)

    logger.info("Pipeline finished for models: {}", ", ".join(models))


if __name__ == "__main__":
    main()
