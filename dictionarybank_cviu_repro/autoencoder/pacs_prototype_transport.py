import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from autoencoder.pacs_ras_head import content_codes, load_factor_model


def accumulate(cache, indices, streams, model, device, batch_size):
    labels = np.asarray(cache["labels"])
    domains = np.asarray(cache["domains"])
    domain_sums = {name: {} for name in streams}
    domain_counts = {}
    class_sums = {name: {} for name in streams}
    class_counts = {}
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        tensor_rows = torch.tensor(rows, dtype=torch.long)
        data = {
            name: cache[name][tensor_rows].to(device) for name in streams
        }
        codes = content_codes(model, data)
        for name in streams:
            values = codes[name].float().cpu()
            for row, value in zip(rows, values):
                domain = str(domains[row])
                label = int(labels[row])
                domain_sums[name].setdefault(
                    domain, torch.zeros(value.shape[0])
                )
                domain_sums[name][domain] += value
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                key = (label, domain)
                class_sums[name].setdefault(
                    key, torch.zeros(value.shape[0])
                )
                class_sums[name][key] += value
                class_counts[key] = class_counts.get(key, 0) + 1
    domain_means = {
        name: {
            domain: value / domain_counts[domain]
            for domain, value in domain_sums[name].items()
        }
        for name in streams
    }
    class_means = {
        name: {
            key: value / class_counts[key]
            for key, value in class_sums[name].items()
        }
        for name in streams
    }
    return domain_means, class_means


def target_codes(cache, indices, streams, model, device, batch_size):
    result = {name: [] for name in streams}
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        data = {
            name: cache[name][rows].to(device) for name in streams
        }
        codes = content_codes(model, data)
        for name in streams:
            result[name].append(codes[name].float().cpu())
    return {name: torch.cat(values) for name, values in result.items()}


def build_prototypes(
    domain_means,
    class_means,
    source_domains,
    classes,
    rank,
    centered,
):
    means = torch.stack([
        domain_means[domain] for domain in source_domains
    ])
    global_mean = means.mean(dim=0)
    prototype_values = []
    for label in range(classes):
        values = [
            class_means[(label, domain)]
            - (domain_means[domain] if centered else 0.0)
            for domain in source_domains
            if (label, domain) in class_means
        ]
        if not values:
            prototype_values.append(torch.zeros_like(global_mean))
        else:
            prototype_values.append(torch.stack(values).mean(0))
    prototypes = torch.stack(prototype_values)
    if centered and rank > 0:
        shifts = means - global_mean
        left, _, _ = torch.linalg.svd(
            shifts.T, full_matrices=False
        )
        basis = left[:, :min(rank, left.shape[1])].T
        prototypes = prototypes - (prototypes @ basis.T) @ basis
    return F.normalize(prototypes, dim=1), global_mean


def evaluate_setting(
    source_domain_means,
    source_class_means,
    source_domains,
    pseudo_domain,
    validation,
    labels,
    classes,
    ranks,
    scales,
    stream,
):
    domains = [
        domain for domain in source_domains if domain != pseudo_domain
    ]
    pseudo_mean = source_domain_means[pseudo_domain]
    rows = validation[pseudo_domain][stream]
    device = rows.device
    pseudo_mean = pseudo_mean.to(device)
    results = {}
    for rank in ranks:
        prototypes, global_mean = build_prototypes(
            source_domain_means,
            source_class_means,
            domains,
            classes,
            rank,
            centered=rank >= 0,
        )
        prototypes = prototypes.to(device)
        basis = None
        if rank > 0:
            source_means = torch.stack([
                source_domain_means[domain] for domain in domains
            ])
            shifts = source_means - source_means.mean(0)
            left, _, _ = torch.linalg.svd(
                shifts.T, full_matrices=False
            )
            basis = left[:, :min(rank, left.shape[1])].T
            basis = basis.to(device)
        adjusted = rows - pseudo_mean
        if basis is not None:
            adjusted = adjusted - (adjusted @ basis.T) @ basis
        for scale in scales:
            key = (rank, scale)
            results[key] = float(
                (
                    (
                        F.normalize(adjusted, dim=1)
                        @ prototypes.T
                        * scale
                    ).argmax(1)
                    == labels[pseudo_domain].to(device)
                ).float().mean()
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unlabeled target prototype transport for factor codes."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--factor-checkpoint", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--streams", nargs="+", default=["clip", "siglip", "dino"])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--scales", type=float, nargs="+", default=[5, 10, 20])
    parser.add_argument("--ensemble-temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache = torch.load(args.cache, weights_only=False)
    target_cache = torch.load(args.target_cache, weights_only=False)
    factor, saved = load_factor_model(
        args.factor_checkpoint, cache, args.streams, device
    )
    all_domains = sorted(set(cache["domains"]))
    source_domains = [
        domain for domain in all_domains if domain != args.target_domain
    ]
    source_rows = np.flatnonzero([
        value in source_domains for value in cache["domains"]
    ])
    source = {
        **{name: cache[name] for name in args.streams},
        "labels": cache["labels"],
        "domains": cache["domains"],
    }
    domain_means, class_means = accumulate(
        source, source_rows, args.streams, factor, device, args.batch_size
    )
    source_labels = np.asarray(cache["labels"])
    source_domains_array = np.asarray(cache["domains"])
    validation = {}
    validation_labels = {}
    for domain in source_domains:
        rows = np.flatnonzero(
            source_domains_array[source_rows] == domain
        )
        absolute = source_rows[rows]
        validation[domain] = {
            name: cache[name][torch.tensor(absolute, dtype=torch.long)]
            .to(device)
            for name in args.streams
        }
        encoded = content_codes(factor, validation[domain])
        validation[domain] = encoded
        validation_labels[domain] = torch.tensor(
            source_labels[absolute], dtype=torch.long
        )
    classes = int(np.asarray(cache["labels"]).max()) + 1
    target_rows = np.flatnonzero([
        value == args.target_domain for value in target_cache["domains"]
    ])
    target_encoded = target_codes(
        target_cache,
        torch.tensor(target_rows, dtype=torch.long),
        args.streams,
        factor,
        device,
        args.batch_size,
    )
    target_mean = {
        name: target_encoded[name].mean(0) for name in args.streams
    }
    target_labels = torch.tensor(
        np.asarray(target_cache["labels"])[target_rows],
        dtype=torch.long,
    )
    report = {
        "settings": vars(args),
        "factor_settings": saved,
        "target_domain": args.target_domain,
        "target_rows": len(target_rows),
        "streams": {},
    }
    probabilities = []
    stream_validation_scores = []
    for name in args.streams:
        candidates = {}
        for pseudo in source_domains:
            result = evaluate_setting(
                domain_means[name],
                class_means[name],
                source_domains,
                pseudo,
                validation,
                validation_labels,
                classes,
                args.ranks,
                args.scales,
                name,
            )
            for key, value in result.items():
                candidates.setdefault(key, []).append(value)
        validation_scores = {
            key: float(np.mean(values))
            for key, values in candidates.items()
        }
        selected_rank, selected_scale = max(
            validation_scores, key=validation_scores.get
        )
        prototypes, global_mean = build_prototypes(
            domain_means[name],
            class_means[name],
            source_domains,
            classes,
            selected_rank,
            centered=True,
        )
        prototypes = prototypes.to(target_encoded[name].device)
        basis = None
        if selected_rank > 0:
            source_means = torch.stack([
                domain_means[name][domain] for domain in source_domains
            ])
            shifts = source_means - source_means.mean(0)
            left, _, _ = torch.linalg.svd(
                shifts.T, full_matrices=False
            )
            basis = left[:, :min(selected_rank, left.shape[1])].T
            basis = basis.to(target_encoded[name].device)
        adjusted = target_encoded[name] - target_mean[name]
        if basis is not None:
            adjusted = adjusted - (adjusted @ basis.T) @ basis
        logits = (
            F.normalize(adjusted, dim=1)
            @ prototypes.T
            * selected_scale
        )
        probability = F.softmax(logits, dim=1)
        probabilities.append(probability)
        stream_validation_scores.append(
            validation_scores[(selected_rank, selected_scale)]
        )
        report["streams"][name] = {
            "selected_rank": selected_rank,
            "selected_scale": selected_scale,
            "validation_accuracy": validation_scores[
                (selected_rank, selected_scale)
            ],
            "target_accuracy": float(
                (probability.argmax(1) == target_labels).float().mean()
            ),
        }
    ensemble = torch.stack(probabilities).mean(0)
    report["ensemble_accuracy"] = float(
        (ensemble.argmax(1) == target_labels).float().mean()
    )
    stream_weights = torch.softmax(
        torch.tensor(
            stream_validation_scores,
            dtype=torch.float32,
        )
        / args.ensemble_temperature,
        dim=0,
    )
    weighted = torch.stack([
        weight * probability
        for weight, probability in zip(stream_weights, probabilities)
    ]).sum(0)
    report["stream_validation_scores"] = {
        name: value
        for name, value in zip(args.streams, stream_validation_scores)
    }
    report["stream_weights"] = {
        name: float(value)
        for name, value in zip(args.streams, stream_weights)
    }
    report["weighted_ensemble_accuracy"] = float(
        (weighted.argmax(1) == target_labels).float().mean()
    )
    torch.save(
        {
            "args": vars(args),
            "factor_settings": saved,
            "ensemble_probabilities": ensemble.cpu(),
        },
        args.out,
    )
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
