import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def accuracy(probability, labels):
    return float(probability.argmax(1).eq(labels).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["clip", "siglip", "dino"],
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    target_cache = torch.load(
        args.target_cache,
        map_location="cpu",
        weights_only=False,
    )
    domains = np.asarray(cache["domains"])
    labels_np = np.asarray(cache["labels"])
    source_domains = sorted(
        value for value in set(domains.tolist()) if value != args.target_domain
    )
    source_rows = np.flatnonzero(
        [value in source_domains for value in domains]
    )
    target_rows = np.flatnonzero(
        np.asarray(target_cache["domains"]) == args.target_domain
    )
    target_labels = torch.tensor(
        np.asarray(target_cache["labels"])[target_rows],
        dtype=torch.long,
    )
    classes = int(labels_np.max()) + 1

    probabilities = []
    stream_report = {}
    for name in args.streams:
        source = F.normalize(cache[name][source_rows].float(), dim=1)
        source_labels = torch.tensor(labels_np[source_rows], dtype=torch.long)
        source_domain_values = domains[source_rows].tolist()
        domain_names = source_domains
        domain_means = []
        centered_classes = []
        for domain in domain_names:
            domain_rows = np.flatnonzero(
                np.asarray(source_domain_values) == domain
            )
            domain_mean = source[domain_rows].mean(0)
            domain_means.append(domain_mean)
            valid_classes = []
            for label in range(classes):
                rows = domain_rows[source_labels[domain_rows].numpy() == label]
                if len(rows):
                    valid_classes.append(
                        source[rows].mean(0) - domain_mean
                    )
                else:
                    valid_classes.append(torch.zeros_like(domain_mean))
            centered_classes.append(torch.stack(valid_classes))
        prototypes = F.normalize(
            torch.stack(centered_classes).mean(0),
            dim=1,
        )
        center = torch.stack(domain_means).mean(0)
        target = F.normalize(
            target_cache[name][target_rows].float(),
            dim=1,
        )
        query = F.normalize(target - center[None], dim=1)
        probability = F.softmax(query @ prototypes.T, dim=1)
        probabilities.append(probability)
        stream_report[name] = {
            "target_accuracy": accuracy(probability, target_labels)
        }

    ensemble = torch.stack(probabilities).mean(0)
    report = {
        "target_domain": args.target_domain,
        "target_rows": len(target_rows),
        "protocol": "l2_raw_source_centered_prototypes",
        "streams": stream_report,
        "target_accuracy": accuracy(ensemble, target_labels),
    }
    if not 0.0 <= report["target_accuracy"] <= 1.0:
        raise AssertionError("invalid raw accuracy")
    torch.save(
        {
            "target_probability": ensemble.cpu(),
            "target_labels": target_labels,
        },
        args.out,
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
