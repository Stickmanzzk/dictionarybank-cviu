"""Analyze RVL bank-composition and paired test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest


METHODS = (
    "single_seed0",
    "single_seed1",
    "single_seed2",
    "bank_01",
    "bank_02",
    "bank_12",
    "bank_full",
    "repeated_seed0_top192",
    "wide192",
)


def load_payload(root: Path, method: str) -> dict:
    return torch.load(
        root / "eval" / f"{method}.pt",
        map_location="cpu",
        weights_only=False,
    )


def correctness(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    probability = payload["target_probability"].float()
    labels = payload["target_labels"].long()
    if len(probability) != len(labels):
        raise AssertionError("probability/label length mismatch")
    if not torch.isfinite(probability).all():
        raise AssertionError("non-finite probability")
    return probability.argmax(1).eq(labels), labels


def paired_compare(
    left: torch.Tensor,
    right: torch.Tensor,
    samples: int,
    seed: int,
) -> dict:
    delta = left.float() - right.float()
    generator = np.random.default_rng(seed)
    values = np.empty(samples)
    delta_np = delta.numpy()
    for index in range(samples):
        rows = generator.integers(0, len(delta_np), len(delta_np))
        values[index] = delta_np[rows].mean()
    left_only = int((left & ~right).sum())
    right_only = int((right & ~left).sum())
    discordant = left_only + right_only
    return {
        "delta_points": 100.0 * float(delta.mean()),
        "bootstrap_95_ci_points": (
            100.0 * np.quantile(values, [0.025, 0.975])
        ).tolist(),
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "mcnemar_exact_p": (
            float(
                binomtest(
                    left_only,
                    discordant,
                    0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    payloads = {
        method: load_payload(args.root, method)
        for method in METHODS
    }
    predictions = {}
    labels = None
    for method, payload in payloads.items():
        predictions[method], current_labels = correctness(payload)
        if labels is None:
            labels = current_labels
        elif not torch.equal(labels, current_labels):
            raise AssertionError(f"label mismatch: {method}")

    accuracies = {
        method: float(predictions[method].float().mean())
        for method in METHODS
    }
    single_values = [
        accuracies[f"single_seed{seed}"] for seed in range(3)
    ]
    pair_values = [
        accuracies[f"bank_{name}"] for name in ("01", "02", "12")
    ]

    comparisons = {}
    seeds = {
        "single_seed0": 1001,
        "single_seed1": 1002,
        "single_seed2": 1003,
        "wide192": 1004,
        "repeated_seed0_top192": 1005,
    }
    for baseline, seed in seeds.items():
        comparisons[f"bank_full_vs_{baseline}"] = paired_compare(
            predictions["bank_full"],
            predictions[baseline],
            args.bootstrap_samples,
            seed,
        )

    per_bank = {}
    for bank in ("bank_01", "bank_02", "bank_12", "bank_full"):
        per_bank[bank] = {
            "accuracy": accuracies[bank],
            "per_single": {
                method: paired_compare(
                    predictions[bank],
                    predictions[method],
                    args.bootstrap_samples,
                    2000 + index,
                )
                for index, method in enumerate(
                    ("single_seed0", "single_seed1", "single_seed2")
                )
            },
        }

    summary = {
        "run": args.root.name,
        "rows": int(len(labels)),
        "accuracies": accuracies,
        "single_seed": {
            "mean": float(np.mean(single_values)),
            "std": float(np.std(single_values, ddof=1)),
            "min": float(min(single_values)),
            "max": float(max(single_values)),
            "values": single_values,
        },
        "pairwise_bank": {
            "mean": float(np.mean(pair_values)),
            "std": float(np.std(pair_values, ddof=1)),
            "min": float(min(pair_values)),
            "max": float(max(pair_values)),
            "values": pair_values,
        },
        "comparisons": comparisons,
        "per_bank": per_bank,
        "full_bank_above_weakest_single": (
            accuracies["bank_full"] > min(single_values)
        ),
    }

    if not summary["full_bank_above_weakest_single"]:
        raise AssertionError("full bank does not exceed the weakest single seed")
    if not all(0.0 <= value <= 1.0 for value in accuracies.values()):
        raise AssertionError("accuracy outside [0, 1]")

    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
