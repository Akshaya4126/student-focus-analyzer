"""CLI entry point for training and saving the focus classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

from model import make_sample_dataset, train_and_save

ROOT = Path(__file__).resolve().parent


def ensure_artifacts(dataset_path=ROOT / "dataset.csv", model_path=ROOT / "model.joblib", metrics_path=ROOT / "model_metrics.json"):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        make_sample_dataset().to_csv(dataset_path, index=False)
    if not Path(model_path).exists() or not Path(metrics_path).exists():
        return train_and_save(dataset_path, model_path, metrics_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train the student focus classification model.")
    parser.add_argument("--dataset", default=str(ROOT / "dataset.csv"))
    parser.add_argument("--model", default=str(ROOT / "model.joblib"))
    parser.add_argument("--metrics", default=str(ROOT / "model_metrics.json"))
    args = parser.parse_args()
    result = train_and_save(args.dataset, args.model, args.metrics)
    print(f"Model trained. Hold-out accuracy: {result['accuracy']:.1%} on {result['test_rows']} test rows.")
    print(f"Dataset: {args.dataset}\nModel: {args.model}\nMetrics: {args.metrics}")


if __name__ == "__main__":
    main()
