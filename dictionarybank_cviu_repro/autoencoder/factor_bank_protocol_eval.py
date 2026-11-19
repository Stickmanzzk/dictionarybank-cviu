import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from autoencoder.fused_factor_bank import (
    build_models,
    code_mean,
    prototypes_from_means,
)
from autoencoder.pacs_factor_sae import split_source_rows


def accuracy(probability, labels):
    return float(
        (probability.argmax(1) == labels).float().mean()
    )


def accumulate_means(
    models,
    cache,
    rows,
    streams,
    device,
    batch_size,
):
    rows = np.asarray(rows)
    labels_all = np.asarray(cache["labels"], dtype=np.int64)
    domains_all = np.asarray(cache["domains"])
    domains = sorted(set(domains_all[rows].tolist()))
    domain_index = {value: index for index, value in enumerate(domains)}
    classes = int(labels_all.max()) + 1
    groups = int(models[streams[0]].groups)
    latent = int(models[streams[0]].latent)
    width = groups * latent

    domain_sums = {
        name: torch.zeros(len(domains), width, device=device)
        for name in streams
    }
    domain_counts = torch.zeros(len(domains), device=device)
    pair_sums = {
        name: torch.zeros(
            classes * len(domains), width, device=device
        )
        for name in streams
    }
    pair_counts = torch.zeros(classes * len(domains), device=device)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = torch.tensor(batch_rows)
        batch_domains = torch.tensor(
            [domain_index[domains_all[row]] for row in batch_rows],
            device=device,
        )
        batch_labels = torch.tensor(
            labels_all[batch_rows], device=device
        )
        pair_ids = batch_labels * len(domains) + batch_domains
        domain_counts.index_add_(
            0,
            batch_domains,
            torch.ones(len(batch_rows), device=device),
        )
        pair_counts.index_add_(
            0,
            pair_ids,
            torch.ones(len(batch_rows), device=device),
        )
        for name in streams:
            values = models[name](cache[name][batch].to(device))
            values = values.float().reshape(len(batch_rows), width)
            domain_sums[name].index_add_(0, batch_domains, values)
            pair_sums[name].index_add_(0, pair_ids, values)

    domain_means = {
        name: domain_sums[name] / domain_counts.clamp_min(1).unsqueeze(1)
        for name in streams
    }
    pair_means = {
        name: pair_sums[name] / pair_counts.clamp_min(1).unsqueeze(1)
        for name in streams
    }
    return domains, domain_means, pair_means, pair_counts, classes, width


def make_prototypes(
    domain_means,
    pair_means,
    pair_counts,
    source_domains,
    domain_index,
    classes,
    width,
):
    output = torch.zeros(classes, width, device=pair_means.device)
    counts = torch.zeros(classes, device=pair_means.device)
    for domain in source_domains:
        domain_id = domain_index[domain]
        domain_batch = torch.full(
            (classes,),
            domain_id,
            dtype=torch.long,
            device=pair_means.device,
        )
        pair_ids = torch.arange(classes, device=pair_means.device)
        pair_ids = pair_ids * len(domain_index) + domain_batch
        valid = pair_counts[pair_ids] > 0
        centered = (
            pair_means[pair_ids]
            - domain_means[domain_id][None]
        )
        output.add_(centered * valid.float()[:, None])
        counts.add_(valid.float())
    return F.normalize(output / counts.clamp_min(1)[:, None], dim=1)


def group_probabilities(
    models,
    cache,
    rows,
    streams,
    prototypes,
    centers,
    device,
    batch_size,
):
    groups = int(models[streams[0]].groups)
    latent = int(models[streams[0]].latent)
    result = {
        name: torch.empty(
            len(rows), groups, prototypes[name].shape[0]
        )
        for name in streams
    }
    for start in range(0, len(rows), batch_size):
        batch_rows = np.asarray(rows[start:start + batch_size])
        batch = torch.tensor(batch_rows)
        for name in streams:
            codes = models[name](cache[name][batch].to(device))
            for group in range(groups):
                values = F.normalize(
                    codes[:, group, :]
                    - centers[name][group * latent:(group + 1) * latent],
                    dim=1,
                )
                group_prototypes = prototypes[name][
                    :, group * latent:(group + 1) * latent
                ]
                result[name][start:start + len(batch_rows), group] = (
                    F.softmax(values @ group_prototypes.T, dim=1).cpu()
                )
    return result


def source_center(models, cache, rows, streams, domains, device, batch_size):
    result = {}
    for name in streams:
        sums = {}
        counts = {}
        domain_values = np.asarray(cache["domains"])
        for start in range(0, len(rows), batch_size):
            batch_rows = np.asarray(rows[start:start + batch_size])
            batch = torch.tensor(batch_rows)
            values = models[name](cache[name][batch].to(device))
            values = values.float().reshape(len(batch_rows), -1).cpu()
            for local, row in enumerate(batch_rows):
                domain = str(domain_values[row])
                sums.setdefault(domain, torch.zeros(values.shape[1]))
                sums[domain] += values[local]
                counts[domain] = counts.get(domain, 0) + 1
        result[name] = torch.stack([
            sums[domain] / counts[domain] for domain in domains
        ]).mean(0).to(device)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="FactorBank source, target and streaming center protocols."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--streams", nargs="+", default=["clip", "siglip", "dino"]
    )
    parser.add_argument(
        "--center-mode",
        choices=("target", "source", "streaming"),
        default="target",
    )
    parser.add_argument("--streaming-decay", type=float, default=0.9)
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
        domain for domain in all_domains if domain != args.target_domain
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
    fitting, validation = split_source_rows(
        source, source_domains, 0
    )
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

    models, groups, latent, group_k = build_models(
        args.checkpoints, args.streams, device
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
        domain: index for index, domain in enumerate(domains)
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
    source_centers = source_center(
        models,
        cache,
        source_rows,
        args.streams,
        domains,
        device,
        args.eval_batch_size,
    )

    if args.center_mode == "source":
        centers = source_centers
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
    elif args.center_mode == "target":
        centers = code_mean(
            models,
            target,
            target_rows,
            args.streams,
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
    else:
        centers = {
            name: value.clone() for name, value in source_centers.items()
        }
        probability = {
            name: torch.empty(
                len(target_rows), groups, classes
            )
            for name in args.streams
        }
        for start in range(0, len(target_rows), args.eval_batch_size):
            batch_rows = target_rows[start:start + args.eval_batch_size]
            batch_probability = group_probabilities(
                models,
                target,
                batch_rows,
                args.streams,
                prototypes,
                centers,
                device,
                args.eval_batch_size,
            )
            for name in args.streams:
                probability[name][start:start + len(batch_rows)] = (
                    batch_probability[name]
                )
            batch_center = code_mean(
                models,
                target,
                batch_rows,
                args.streams,
                device,
                args.eval_batch_size,
            )
            for name in args.streams:
                centers[name] = (
                    args.streaming_decay * centers[name]
                    + (1.0 - args.streaming_decay) * batch_center[name]
                )

    target_probability = torch.stack([
        probability[name][:, group, :]
        for name in args.streams
        for group in range(groups)
    ]).mean(0)
    report = {
        "settings": vars(args),
        "target_domain": args.target_domain,
        "target_rows": len(target_rows),
        "groups": groups,
        "group_k": group_k,
        "streams": {
            name: {
                "target_accuracy": accuracy(
                    probability[name].mean(1), target_labels
                ),
                "members": [
                    accuracy(probability[name][:, group], target_labels)
                    for group in range(groups)
                ],
            }
            for name in args.streams
        },
        "target_accuracy": accuracy(target_probability, target_labels),
    }
    torch.save(
        {
            "args": vars(args),
            "target_probability": target_probability.cpu(),
            "member_probability": {
                name: value.cpu()
                for name, value in probability.items()
            },
            "target_labels": target_labels.cpu(),
        },
        args.out,
    )
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
