"""Summarize a saved AMFormer attention matrix without retraining."""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "results" / "attention" / "attention_layer_0.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "attention" / "feature_attention_summary.csv"


def summarize_attention(input_path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Rank features by mean attention received across the saved matrix."""
    matrix = pd.read_csv(input_path, index_col=0)
    summary = matrix.mean(axis=0).rename("mean_attention").sort_values(ascending=False)
    return summary.rename_axis("feature").reset_index()


def main() -> None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summarize_attention().to_csv(DEFAULT_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved attention summary to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
