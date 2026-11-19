import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json

import numpy as np
import torch

from autoencoder.multistream_sae_train import MultiStreamSAE
from experiment_utils import metrics
from factor_eval import variance_decomposition


WIDTHS = {"clip": 768, "siglip": 768, "dino": 1024}
DOMAINS = {
    "documents": (
        "documents_clip_openai.pt",
        "documents_siglip.pt",
        "documents_dinov2_l.pt",
    ),
    "shapes": (
        "shapes_clip_openai.pt",
        "shapes_siglip.pt",
        "shapes_dinov2_l.pt",
    ),
    "real scans": (
        "real_scans_funsd_clip_openai.pt",
        "real_scans_funsd_siglip.pt",
        "real_scans_funsd_dinov2_l.pt",
    ),
    "arxiv": (
        "real_scans_wide_clip_openai.pt",
        "real_scans_wide_siglip.pt",
        "real_scans_wide_dinov2_l.pt",
    ),
}


def build_codes(logits, scores, k):
    names = list(logits)
    _, indices = torch.topk(torch.cat(scores, dim=1), k, dim=1)
    codes, offset = {}, 0
    for name in names:
        width = logits[name].shape[1]
        local = (indices - offset).clamp(0, width - 1)
        mask = (indices >= offset) & (indices < offset + width)
        raw = torch.relu(logits[name].gather(1, local))
        codes[name] = torch.zeros_like(logits[name]).scatter_(
            1, local, torch.where(mask, raw, torch.zeros_like(raw))
        )
        offset += width
    return codes


def scores(logits, mode):
    names = list(WIDTHS)
    if mode == "raw":
        return [logits[name] for name in names]
    if mode == "z":
        return [
            (logits[name] - logits[name].mean(1, keepdim=True))
            / (logits[name].std(1, keepdim=True) + 1e-6)
            for name in names
        ]
    if mode == "robust":
        out = []
        for name in names:
            values = logits[name]
            median = values.median(1, keepdim=True).values
            mad = (values - median).abs().median(1, keepdim=True).values
            out.append((values - median) / (1.4826 * mad + 1e-6))
        return out
    if mode == "rank":
        out = []
        for name in names:
            values = logits[name]
            order = values.argsort(dim=1).argsort(dim=1).float()
            out.append(order / max(1, values.shape[1] - 1))
        return out
    raise ValueError(mode)


def subset_retrieval(codes, contents, layouts, rows):
    rows = np.asarray(rows)
    return metrics(
        codes[rows],
        [contents[int(index)] for index in rows],
        [layouts[int(index)] for index in rows],
    )["NN_R@1"]


def three_way_split(contents, seed=0):
    unique = np.asarray(sorted(set(contents)))
    order = np.random.default_rng(seed).permutation(len(unique))
    first = len(unique) // 2
    second = 3 * len(unique) // 4
    groups = (
        set(unique[order[:first]].tolist()),
        set(unique[order[first:second]].tolist()),
        set(unique[order[second:]].tolist()),
    )
    masks = [
        np.array([value in group for value in contents])
        for group in groups
    ]
    return [np.flatnonzero(mask) for mask in masks]


def quota_codes(logits, quotas):
    names = list(WIDTHS)
    codes = {}
    for name, quota in zip(names, quotas):
        values, indices = torch.topk(logits[name], quota, dim=1)
        codes[name] = torch.zeros_like(logits[name]).scatter_(
            1, indices, torch.relu(values)
        )
    return codes


def evaluate_method(codes, logits, contents, layouts, rows):
    return {
        name: {
            "R@1": subset_retrieval(codes[name], contents, layouts, rows),
            "slots": float((codes[name][rows] > 0).sum(1).float().mean()),
            "mass": float(codes[name][rows].sum()),
        }
        for name in WIDTHS
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare calibration operators for shared top-k gates."
    )
    parser.add_argument("--checkpoint", default="outputs/visual3_sae.pt")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--report-json", default="outputs/gate_variants.json"
    )
    args = parser.parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    blob = torch.load(args.checkpoint, weights_only=False)
    model = MultiStreamSAE(WIDTHS, 4096, args.k).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    report = {}
    quota_grid = range(8, 49, 4)
    for domain, files in DOMAINS.items():
        caches = [
            torch.load(os.path.join(args.cache_dir, path), weights_only=False)
            for path in files
        ]
        contents = caches[0]["contents"]
        layouts = caches[0]["layouts"]
        streams = {
            name: caches[index]["features"].to(device)
            for index, name in enumerate(WIDTHS)
        }
        with torch.no_grad():
            logits, _, _ = model(streams)
        logits = {name: value.float().cpu() for name, value in logits.items()}
        fitting, calibration, test = three_way_split(contents, seed=0)
        row = {"methods": {}}
        for mode in ("raw", "z", "robust", "rank"):
            codes = build_codes(logits, scores(logits, mode), args.k)
            row["methods"][mode] = evaluate_method(
                codes, logits, contents, layouts, test
            )
        per_stream = {
            name: torch.zeros_like(logits[name]).scatter_(
                1,
                torch.topk(logits[name], args.k, dim=1).indices,
                torch.relu(torch.topk(logits[name], args.k, dim=1).values),
            )
            for name in WIDTHS
        }
        row["methods"]["per_stream"] = evaluate_method(
            per_stream, logits, contents, layouts, test
        )

        rank_scores = scores(logits, "rank")
        rank_candidates = []
        for clip_weight in (0.25, 0.5, 1, 2, 4, 8, 16):
            for siglip_weight in (0.25, 0.5, 1, 2, 4, 8, 16):
                weighted_scores = [
                    clip_weight * rank_scores[0],
                    siglip_weight * rank_scores[1],
                    rank_scores[2],
                ]
                codes = build_codes(logits, weighted_scores, args.k)
                values = [
                    subset_retrieval(
                        codes[name], contents, layouts, calibration
                    )
                    for name in WIDTHS
                ]
                slots = [
                    float((codes[name] > 0).sum(1).float().mean())
                    for name in WIDTHS
                ]
                rank_candidates.append({
                    "weights": {
                        "clip": clip_weight,
                        "siglip": siglip_weight,
                        "dino": 1.0,
                    },
                    "content": values,
                    "slots": slots,
                    "objective": float(np.mean(values)),
                    "min_content": float(np.min(values)),
                })
        rank_best = max(
            rank_candidates,
            key=lambda candidate:
            (candidate["min_content"], candidate["objective"]),
        )
        weighted_scores = [
            rank_best["weights"]["clip"] * rank_scores[0],
            rank_best["weights"]["siglip"] * rank_scores[1],
            rank_scores[2],
        ]
        row["rank_selected_weights"] = rank_best["weights"]
        row["methods"]["rank_weighted"] = evaluate_method(
            build_codes(logits, weighted_scores, args.k),
            logits,
            contents,
            layouts,
            test,
        )
        fitting_contents = [contents[int(index)] for index in fitting]
        fitting_layouts = [layouts[int(index)] for index in fitting]
        utility = []
        for name in WIDTHS:
            nuisance_var, content_var = variance_decomposition(
                logits[name][fitting],
                fitting_contents,
                fitting_layouts,
            )
            values = torch.log(
                (content_var + 1e-6) / (nuisance_var + 1e-6)
            )
            utility.append(
                (values - values.mean()) / (values.std() + 1e-6)
            )
        utility_candidates = []
        for strength in (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5):
            prior_scores = [
                rank_scores[index] + strength * utility[index]
                for index in range(len(WIDTHS))
            ]
            codes = build_codes(logits, prior_scores, args.k)
            result = evaluate_method(
                codes,
                logits,
                contents,
                layouts,
                test,
            )
            row["methods"][f"rank_utility={strength:g}"] = result
            calibration_values = [
                subset_retrieval(codes[name], contents, layouts, calibration)
                for name in WIDTHS
            ]
            utility_candidates.append({
                "strength": strength,
                "content": calibration_values,
                "objective": float(np.mean(calibration_values)),
            })
        utility_best = max(
            utility_candidates,
            key=lambda candidate: candidate["objective"],
        )
        selected_scores = [
            rank_scores[index]
            + utility_best["strength"] * utility[index]
            for index in range(len(WIDTHS))
        ]
        row["utility_strength"] = utility_best["strength"]
        row["methods"]["utility_calibrated"] = evaluate_method(
            build_codes(logits, selected_scores, args.k),
            logits,
            contents,
            layouts,
            test,
        )

        candidates = []
        for clip_quota in range(8, 49, 4):
            for siglip_quota in range(8, 49, 4):
                dino_quota = args.k - clip_quota - siglip_quota
                if dino_quota < 8:
                    continue
                codes = quota_codes(
                    logits, (clip_quota, siglip_quota, dino_quota)
                )
                values = [
                    subset_retrieval(
                        codes[name], contents, layouts, calibration
                    )
                    for name in WIDTHS
                ]
                candidates.append({
                    "quotas": (clip_quota, siglip_quota, dino_quota),
                    "utility": float(np.mean(values)),
                    "min_utility": float(np.min(values)),
                })
        chosen = max(
            candidates,
            key=lambda candidate:
            (candidate["min_utility"], candidate["utility"]),
        )
        quota = quota_codes(logits, chosen["quotas"])
        row["selected_quotas"] = chosen
        row["methods"]["quota"] = evaluate_method(
            quota, logits, contents, layouts, test
        )
        report[domain] = row
        print(
            f"{domain:11s} rank_w={row['rank_selected_weights']} "
            f"utility_lambda={row['utility_strength']} "
            f"quota={chosen['quotas']} "
            + " ".join(
                f"{mode}:" + "/".join(
                    f"{row['methods'][mode][name]['R@1']:.2f}"
                    for name in WIDTHS
                )
                for mode in (
                    "raw", "z", "robust", "rank", "rank_weighted",
                    "utility_calibrated", "per_stream", "quota"
                )
            )
        )
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
