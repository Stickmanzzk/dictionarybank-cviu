import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from autoencoder.factor_bank_protocol_eval import (  # noqa: E402
    accumulate_means,
    group_probabilities,
    make_prototypes,
    source_center,
)
from autoencoder.pacs_factor_sae import split_source_rows  # noqa: E402


class BudgetBank(torch.nn.Module):
    def __init__(self, width, members, latent, active_budget):
        super().__init__()
        self.groups = 1
        self.latent = members * latent
        self.group_k = active_budget
        self.weight = torch.nn.Parameter(
            torch.empty(self.latent, width), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self.latent), requires_grad=False
        )

    def forward(self, values):
        logits = F.linear(values, self.weight, self.bias)
        selected, indices = torch.topk(
            logits, self.group_k, dim=1
        )
        code = torch.zeros_like(logits).scatter_(
            1, indices, torch.relu(selected)
        )
        return code[:, None, :]


def build_models(paths, streams, active_budget, device):
    blobs = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    saved = blobs[0]["args"]
    members = len(blobs)
    latent = int(saved["content_latents"])
    if active_budget > members * latent:
        raise ValueError("active budget exceeds bank width")
    models = {}
    for name in streams:
        weights = []
        biases = []
        for blob in blobs:
            state = blob["state_dict"]
            weight = state[f"content_encoders.{name}.weight"].float()
            bias = state[f"content_encoders.{name}.bias"].float()
            pre_bias = state[f"pre_bias.{name}"].float()
            weights.append(weight)
            biases.append(bias - weight @ pre_bias)
        model = BudgetBank(
            weights[0].shape[1], members, latent, active_budget
        )
        with torch.no_grad():
            model.weight.copy_(torch.cat(weights, 0))
            model.bias.copy_(torch.cat(biases, 0))
        models[name] = model.to(device).eval()
    return models


def accuracy(probability, labels):
    return float(probability.argmax(1).eq(labels).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--streams", nargs="+", default=["clip", "siglip", "dino"]
    )
    parser.add_argument("--active-budget", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache = torch.load(args.cache, weights_only=False)
    target_cache = torch.load(args.target_cache, weights_only=False)
    all_domains = sorted(set(cache["domains"]))
    source_domains = [
        value for value in all_domains if value != args.target_domain
    ]
    source_rows = np.flatnonzero([
        value in source_domains for value in cache["domains"]
    ])
    source = {
        **{name: cache[name] for name in args.streams},
        "labels": np.asarray(cache["labels"])[source_rows],
        "domains": [
            cache["domains"][int(row)] for row in source_rows
        ],
    }
    split_source_rows(source, source_domains, 0)
    target_rows = np.flatnonzero([
        value == args.target_domain
        for value in target_cache["domains"]
    ])
    target = {
        name: target_cache[name] for name in args.streams
    }
    target_labels = torch.tensor(
        np.asarray(target_cache["labels"])[target_rows],
        dtype=torch.long,
    )
    models = build_models(
        args.checkpoints,
        args.streams,
        args.active_budget,
        device,
    )
    domains, domain_means, pair_means, pair_counts, classes, width = (
        accumulate_means(
            models,
            cache,
            source_rows,
            args.streams,
            device,
            args.eval_batch_size,
        )
    )
    domain_index = {
        value: index for index, value in enumerate(domains)
    }
    prototypes = {
        name: make_prototypes(
            domain_means[name],
            pair_means[name],
            pair_counts,
            source_domains,
            domain_index,
            classes,
            width,
        )
        for name in args.streams
    }
    centers = source_center(
        models,
        cache,
        source_rows,
        args.streams,
        domains,
        device,
        args.eval_batch_size,
    )
    probability = group_probabilities(
        models,
        target,
        target_rows,
        args.streams,
        prototypes,
        centers,
        device,
        args.eval_batch_size,
    )
    per_stream = {
        name: torch.stack([
            probability[name][:, group, :]
            for group in range(probability[name].shape[1])
        ]).mean(0)
        for name in args.streams
    }
    target_probability = torch.stack(
        list(per_stream.values())
    ).mean(0)
    report = {
        "settings": vars(args),
        "target_domain": args.target_domain,
        "target_accuracy": accuracy(
            target_probability, target_labels
        ),
        "streams": {
            name: {
                "target_accuracy": accuracy(
                    value, target_labels
                )
            }
            for name, value in per_stream.items()
        },
    }
    torch.save(
        {
            "args": vars(args),
            "target_probability": target_probability,
            "target_labels": target_labels,
        },
        args.out,
    )
    Path(args.report_json).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
