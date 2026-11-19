import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def correct(payload):
    probability = payload["target_probability"].float()
    labels = payload["target_labels"].long()
    if len(probability) != len(labels):
        raise AssertionError("prediction/label mismatch")
    return probability.argmax(1).eq(labels), labels


def paired(left, right, samples, seed):
    delta = left.float() - right.float()
    generator = np.random.default_rng(seed)
    values = np.empty(samples)
    raw = delta.numpy()
    for index in range(samples):
        rows = generator.integers(0, len(raw), len(raw))
        values[index] = raw[rows].mean()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    method_names = (
        "single_seed0",
        "single_seed1",
        "single_seed2",
        "bank_01",
        "bank_02",
        "bank_12",
        "bank_full",
        "wide_k64",
        "raw_multistream",
    )
    predictions = {name: {} for name in method_names}
    predictions["single_ensemble"] = {}
    labels = {}
    probabilities = {"single_ensemble": {}}

    for target in range(6):
        single_probabilities = []
        for name in method_names:
            path = args.root / "eval" / f"target{target}_{name}.pt"
            payload = load(path)
            pred, current_labels = correct(payload)
            predictions[name][target] = pred
            if target in labels:
                if not torch.equal(labels[target], current_labels):
                    raise AssertionError(f"label mismatch at target {target}")
            else:
                labels[target] = current_labels
            if name.startswith("single_seed"):
                single_probabilities.append(
                    payload["target_probability"].float()
                )
        ensemble_probability = torch.stack(single_probabilities).mean(0)
        probabilities["single_ensemble"][target] = ensemble_probability
        predictions["single_ensemble"][target] = (
            ensemble_probability.argmax(1).eq(labels[target])
        )

    all_names = method_names + ("single_ensemble",)
    per_target = {}
    for name in all_names:
        per_target[name] = {
            str(target): float(predictions[name][target].float().mean())
            for target in range(6)
        }
    macro = {
        name: float(np.mean(list(values.values())))
        for name, values in per_target.items()
    }

    comparisons = {}
    baselines = (
        "single_seed0",
        "single_seed1",
        "single_seed2",
        "single_ensemble",
        "wide_k64",
        "raw_multistream",
    )
    for index, baseline in enumerate(baselines):
        per_domain = {}
        bank_correct = []
        baseline_correct = []
        for target in range(6):
            left = predictions["bank_full"][target]
            right = predictions[baseline][target]
            bank_correct.append(left)
            baseline_correct.append(right)
            per_domain[str(target)] = paired(
                left,
                right,
                args.bootstrap_samples,
                5000 + index * 10 + target,
            )
        bank_all = torch.cat(bank_correct)
        baseline_all = torch.cat(baseline_correct)
        comparisons[f"bank_full_vs_{baseline}"] = {
            "macro_delta_points": 100.0
            * (macro["bank_full"] - macro[baseline]),
            "pooled": paired(
                bank_all,
                baseline_all,
                args.bootstrap_samples,
                6000 + index,
            ),
            "per_target": per_domain,
        }

    pairs = [
        macro["bank_01"],
        macro["bank_02"],
        macro["bank_12"],
    ]
    singles = [
        macro["single_seed0"],
        macro["single_seed1"],
        macro["single_seed2"],
    ]
    summary = {
        "per_target": per_target,
        "macro": macro,
        "single_seed_stats": {
            "mean": float(np.mean(singles)),
            "std": float(np.std(singles, ddof=1)),
            "min": float(min(singles)),
            "max": float(max(singles)),
        },
        "pairwise_bank_stats": {
            "mean": float(np.mean(pairs)),
            "std": float(np.std(pairs, ddof=1)),
            "min": float(min(pairs)),
            "max": float(max(pairs)),
        },
        "comparisons": comparisons,
    }
    for name, value in macro.items():
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"invalid macro accuracy: {name}")
    (args.root / "ensemble_analysis.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
