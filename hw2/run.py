"""Pipeline entry point to train and evaluate all language models."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from loguru import logger

from evaluate import run_evaluation
from train import run_training
from utils import load_config, plot_multi_model_steps, prepare_logger


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
    histories: dict = {}
    run_counter = 0
    for model_name in models:
        if model_name == "fnn":
            sweep_values = cfg.experiments.fnn_context_sizes
            for context_size in sweep_values:
                run_cfg = copy.deepcopy(cfg)
                run_cfg.fnn.context_size = context_size
                run_label = f"{model_name}_ctx{context_size}"
                gpu_id = None
                if cfg.experiments.cuda_devices:
                    gpu_id = cfg.experiments.cuda_devices[run_counter % len(cfg.experiments.cuda_devices)]
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                    logger.info("Running {} on CUDA_VISIBLE_DEVICES={}", run_label, gpu_id)
                logger.info("========== Training {} ==========", run_label.upper())
                checkpoint_path, history = run_training(
                    model_name,
                    args.config,
                    reuse_logger=True,
                    cfg_override=run_cfg,
                    run_name=run_label,
                )
                histories[run_label] = history
                logger.info("========== Evaluating {} ==========", run_label.upper())
                if checkpoint_path.exists():
                    run_evaluation(
                        model_name,
                        str(checkpoint_path),
                        args.config,
                        reuse_logger=True,
                        cfg_override=run_cfg,
                        run_name=run_label,
                    )
                else:
                    logger.warning("Checkpoint {} missing; skipping evaluation.", checkpoint_path)
                run_counter += 1
        else:
            for max_len in cfg.experiments.max_lengths:
                run_cfg = copy.deepcopy(cfg)
                run_cfg.training.seq_len = max_len
                run_label = f"{model_name}_len{max_len}"
                gpu_id = None
                if cfg.experiments.cuda_devices:
                    gpu_id = cfg.experiments.cuda_devices[run_counter % len(cfg.experiments.cuda_devices)]
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                    logger.info("Running {} on CUDA_VISIBLE_DEVICES={}", run_label, gpu_id)
                logger.info("========== Training {} ==========", run_label.upper())
                checkpoint_path, history = run_training(
                    model_name,
                    args.config,
                    reuse_logger=True,
                    cfg_override=run_cfg,
                    run_name=run_label,
                )
                histories[run_label] = history
                logger.info("========== Evaluating {} ==========", run_label.upper())
                if checkpoint_path.exists():
                    run_evaluation(
                        model_name,
                        str(checkpoint_path),
                        args.config,
                        reuse_logger=True,
                        cfg_override=run_cfg,
                        run_name=run_label,
                    )
                else:
                    logger.warning("Checkpoint {} missing; skipping evaluation.", checkpoint_path)
                run_counter += 1

    plot_multi_model_steps(histories, output_dir)
    logger.info("Pipeline finished for models: {}", ", ".join(models))


if __name__ == "__main__":
    main()
