import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from autoencoder.pacs_factor_sae import split_source_rows
from autoencoder.pacs_prototype_transport import accumulate
from autoencoder.pacs_ras_head import content_codes, load_factor_model


class FusedStreamBank(torch.nn.Module):
    def __init__(self, width, groups, latent, group_k):
        super().__init__()
        self.groups = groups
        self.latent = latent
        self.group_k = group_k
        self.weight = torch.nn.Parameter(
            torch.empty(groups * latent, width), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.empty(groups * latent), requires_grad=False
        )

    def forward(self, values):
        logits = F.linear(values, self.weight, self.bias).view(
            values.shape[0], self.groups, self.latent
        )
        top_values, top_indices = torch.topk(
            logits, self.group_k, dim=2
        )
        return torch.zeros_like(logits).scatter_(
            2, top_indices, torch.relu(top_values)
        )


@torch.no_grad()
def code_mean(models, cache, rows, streams, device, batch_size):
    sums = {
        name: torch.zeros(models[name].weight.shape[0], device=device)
        for name in streams
    }
    count = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = torch.tensor(rows[start:start + batch_size])
        count += len(batch_rows)
        for name in streams:
            values = models[name](
                cache[name][batch_rows].to(device)
            )
            sums[name] += values.float().reshape(
                values.shape[0], -1
            ).sum(0)
    return {
        name: values / count for name, values in sums.items()
    }


@torch.no_grad()
def group_probabilities(
    models,
    cache,
    rows,
    streams,
    prototypes,
    centers,
    groups,
    latent,
    device,
    batch_size,
):
    result = {}
    for name in streams:
        for group in range(groups):
            result[(name, group)] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = torch.tensor(rows[start:start + batch_size])
        for name in streams:
            codes = models[name](
                cache[name][batch_rows].to(device)
            )
            for group in range(groups):
                values = F.normalize(
                    codes[:, group, :]
                    - centers[name][group * latent:(group + 1) * latent],
                    dim=1,
                )
                group_prototypes = prototypes[name][
                    :, group * latent:(group + 1) * latent
                ].to(device)
                result[(name, group)].append(
                    F.softmax(values @ group_prototypes.T, dim=1).cpu()
                )
    return {
        key: torch.cat(values) for key, values in result.items()
    }


def build_models(checkpoint_paths, streams, device):
    blobs = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    saved = blobs[0]["args"]
    groups = len(blobs)
    latent = saved["content_latents"]
    group_k = saved["content_k"]
    models = {}
    for name in streams:
        model = FusedStreamBank(
            saved_width := blobs[0]["state_dict"][
                f"content_encoders.{name}.weight"
            ].shape[1],
            groups,
            latent,
            group_k,
        )
        weights = []
        biases = []
        for blob in blobs:
            state = blob["state_dict"]
            weight = state[f"content_encoders.{name}.weight"]
            bias = state[f"content_encoders.{name}.bias"]
            pre_bias = state[f"pre_bias.{name}"]
            weights.append(weight)
            biases.append(bias - weight @ pre_bias)
        with torch.no_grad():
            model.weight.copy_(torch.cat(weights, 0))
            model.bias.copy_(torch.cat(biases, 0))
        models[name] = model.to(device).eval()
    return models, groups, latent, group_k


def prototypes_from_means(
    class_means,
    domain_means,
    source_domains,
    classes,
):
    result = []
    for label in range(classes):
        values = [
            class_means[(label, domain)] - domain_means[domain]
            for domain in source_domains
            if (label, domain) in class_means
        ]
        result.append(
            torch.stack(values).mean(0)
            if values
            else torch.zeros_like(domain_means[source_domains[0]])
        )
    return F.normalize(torch.stack(result), dim=1)


def accuracy(probability, labels):
    return float(
        probability.argmax(1).eq(labels).float().mean()
    )


def main():
    parser = argparse.ArgumentParser(
        description="One-pass fused content factor bank."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--streams", nargs="+", default=["clip", "siglip", "dino"]
    )
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
        **{name: cache[name][source_rows] for name in args.streams},
        "labels": np.asarray(cache["labels"])[source_rows],
        "domains": [
            cache["domains"][int(row)] for row in source_rows
        ],
    }
    fitting, validation = split_source_rows(
        source, source_domains, 0
    )
    source_labels = torch.tensor(source["labels"], dtype=torch.long)
    classes = int(source_labels.max()) + 1
    target_rows = np.flatnonzero([
        value == args.target_domain
        for value in target_cache["domains"]
    ])
    target = {
        name: target_cache[name][target_rows]
        for name in args.streams
    }
    target_labels = torch.tensor(
        np.asarray(target_cache["labels"])[target_rows],
        dtype=torch.long,
    )
    models, groups, latent, group_k = build_models(
        args.checkpoints, args.streams, device
    )
    fit_domain_means = {name: {} for name in args.streams}
    fit_class_means = {name: {} for name in args.streams}
    full_domain_means = {name: {} for name in args.streams}
    full_class_means = {name: {} for name in args.streams}

    def accumulate_codes(rows):
        domain_sums = {name: {} for name in args.streams}
        domain_counts = {}
        class_sums = {name: {} for name in args.streams}
        class_counts = {}
        for start in range(0, len(rows), args.eval_batch_size):
            batch_rows = torch.tensor(rows[start:start + args.eval_batch_size])
            for name in args.streams:
                values = models[name](
                    source[name][batch_rows].to(device)
                ).float().cpu()
                for local, row in enumerate(batch_rows.tolist()):
                    domain = str(source["domains"][row])
                    label = int(source_labels[row])
                    value = values[local].reshape(-1)
                    domain_sums[name].setdefault(
                        domain, torch.zeros(value.shape)
                    )
                    domain_sums[name][domain] += value
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    key = (label, domain)
                    class_sums[name].setdefault(
                        key, torch.zeros(value.shape)
                    )
                    class_sums[name][key] += value
                    class_counts[key] = class_counts.get(key, 0) + 1
        return (
            {
                name: {
                    domain: value / domain_counts[domain]
                    for domain, value in domain_sums[name].items()
                }
                for name in args.streams
            },
            {
                name: {
                    key: value / class_counts[key]
                    for key, value in class_sums[name].items()
                }
                for name in args.streams
            },
        )

    dm, cm = accumulate_codes(fitting)
    for name in args.streams:
        fit_domain_means[name], fit_class_means[name] = dm[name], cm[name]
    dm, cm = accumulate_codes(np.arange(len(source_labels)))
    for name in args.streams:
        full_domain_means[name], full_class_means[name] = dm[name], cm[name]
    fit_prototypes = {
        name: prototypes_from_means(
            fit_class_means[name],
            fit_domain_means[name],
            source_domains,
            classes,
        )
        for name in args.streams
    }
    full_prototypes = {
        name: prototypes_from_means(
            full_class_means[name],
            full_domain_means[name],
            source_domains,
            classes,
        )
        for name in args.streams
    }
    validation_centers = code_mean(
        models,
        source,
        validation,
        args.streams,
        device,
        args.eval_batch_size,
    )
    target_centers = code_mean(
        models,
        target,
        np.arange(len(target_labels)),
        args.streams,
        device,
        args.eval_batch_size,
    )
    validation_probability = group_probabilities(
        models,
        source,
        validation,
        args.streams,
        fit_prototypes,
        validation_centers,
        groups,
        latent,
        device,
        args.eval_batch_size,
    )
    target_probability = group_probabilities(
        models,
        target,
        np.arange(len(target_labels)),
        args.streams,
        full_prototypes,
        target_centers,
        groups,
        latent,
        device,
        args.eval_batch_size,
    )
    report = {
        "settings": vars(args),
        "target_domain": args.target_domain,
        "target_rows": len(target_labels),
        "groups": groups,
        "group_k": group_k,
        "streams": {},
        "target_accuracy": 0.0,
        "validation_accuracy": 0.0,
    }
    for name in args.streams:
        report["streams"][name] = {}
        for group in range(groups):
            key = (name, group)
            report["streams"][name][group] = {
                "validation_accuracy": accuracy(
                    validation_probability[key],
                    source_labels[validation],
                ),
                "target_accuracy": accuracy(
                    target_probability[key],
                    target_labels,
                ),
            }
    validation_ensemble = torch.stack(
        list(validation_probability.values())
    ).mean(0)
    target_ensemble = torch.stack(
        list(target_probability.values())
    ).mean(0)
    report["validation_accuracy"] = accuracy(
        validation_ensemble, source_labels[validation]
    )
    report["target_accuracy"] = accuracy(
        target_ensemble, target_labels
    )
    report["parameters"] = sum(
        parameter.numel()
        for model in models.values()
        for parameter in model.parameters()
    )
    torch.save({
        "target_probability": target_ensemble,
        "args": vars(args),
    }, args.out)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
