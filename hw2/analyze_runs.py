"""Aggregate finished runs to plot cross-model curves and build a summary table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from utils import load_config, load_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize runs/histories and summarize results.")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config TOML.")
    parser.add_argument(
        "--hist-dir",
        type=str,
        default=None,
        help="Optional override for histories folder (defaults to <output_dir>/histories).",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _summarize_history(run_name: str, history: Dict[str, List[float]]) -> Dict[str, float]:
    val_ppls = history.get("val_ppl", []) or []
    val_losses = history.get("val_loss", []) or []
    train_losses = history.get("train_loss", []) or []
    epochs = history.get("epoch", []) or []
    best_val_ppl = min(val_ppls) if val_ppls else None
    best_epoch = val_ppls.index(best_val_ppl) + 1 if best_val_ppl is not None else None
    return {
        "run": run_name,
        "epochs": len(epochs),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_val_ppl": val_ppls[-1] if val_ppls else None,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
    }


def _write_summary(
    rows: List[Dict[str, float]],
    csv_path: Path,
    md_path: Path,
) -> None:
    if not rows:
        return
    fieldnames = ["run", "epochs", "final_train_loss", "final_val_loss", "final_val_ppl", "best_val_ppl", "best_epoch"]
    _ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in fieldnames})
    def _fmt(value: float | None) -> str:
        return "" if value is None else f"{value:.4f}"

    with md_path.open("w", encoding="utf-8") as f:
        f.write("|run|epochs|final_train_loss|final_val_loss|final_val_ppl|best_val_ppl|best_epoch|\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                "|{run}|{epochs}|{final_train_loss}|{final_val_loss}|{final_val_ppl}|{best_val_ppl}|{best_epoch}|\n".format(
                    run=row["run"],
                    epochs=row["epochs"],
                    final_train_loss=_fmt(row["final_train_loss"]),
                    final_val_loss=_fmt(row["final_val_loss"]),
                    final_val_ppl=_fmt(row["final_val_ppl"]),
                    best_val_ppl=_fmt(row["best_val_ppl"]),
                    best_epoch=row["best_epoch"] or "",
                )
            )


def _select_best_by_model(
    summaries: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    best: Dict[str, Dict[str, float]] = {}
    for row in summaries:
        model_name = row["run"].split("_")[0]
        if model_name not in {"fnn", "rnn", "transformer"}:
            continue
        if row["best_val_ppl"] is None:
            continue
        current = best.get(model_name)
        if current is None or row["best_val_ppl"] < current["best_val_ppl"]:
            best[model_name] = row
    return best


def _plot_multi_model_epochs(
    histories: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure()
    plotted = False
    for run_name, hist in histories.items():
        epochs = hist.get("epoch", [])
        ys = hist.get(metric_key, [])
        if not epochs or not ys:
            continue
        plt.plot(epochs, ys, marker="o", label=run_name.upper())
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def analyze_runs(config_path: str, hist_dir_override: str | None = None) -> Tuple[Path, Path, List[Dict[str, float]]]:
    cfg = load_config(config_path)
    output_root = Path(cfg.paths.output_dir)
    hist_dir = Path(hist_dir_override) if hist_dir_override else output_root / "histories"
    if not hist_dir.exists():
        raise FileNotFoundError(f"History folder not found: {hist_dir}")
    histories: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(hist_dir.glob("*.json")):
        histories[path.stem] = load_history(path)
    if not histories:
        raise RuntimeError(f"No history JSON files found under {hist_dir}")

    summaries = [_summarize_history(name, hist) for name, hist in histories.items()]
    summaries = [row for row in summaries if row["epochs"] > 0]
    summaries.sort(key=lambda r: r["best_val_ppl"] if r["best_val_ppl"] is not None else float("inf"))

    figures_dir = output_root / "figures"
    _ensure_dir(figures_dir)
    best_rows = _select_best_by_model(summaries)
    best_histories = {row["run"]: histories[row["run"]] for row in best_rows.values()}
    _plot_multi_model_epochs(
        best_histories,
        figures_dir,
        metric_key="val_ppl",
        title="Validation Perplexity (Best per Model)",
        ylabel="Perplexity",
        filename="multi_model_val_ppl.png",
    )
    _plot_multi_model_epochs(
        best_histories,
        figures_dir,
        metric_key="val_loss",
        title="Validation Loss (Best per Model)",
        ylabel="Loss",
        filename="multi_model_val_loss.png",
    )

    summary_csv = output_root / "summary.csv"
    summary_md = output_root / "summary.md"
    _write_summary(summaries, summary_csv, summary_md)
    logger.info("Wrote summary table to {} and {}", summary_csv, summary_md)
    logger.info("Saved multi-model comparison plots to {}", figures_dir)
    return summary_csv, summary_md, summaries


def main() -> None:
    args = parse_args()
    analyze_runs(args.config, args.hist_dir)


if __name__ == "__main__":
    main()
