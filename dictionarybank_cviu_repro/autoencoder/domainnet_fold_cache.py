import argparse
import json
import os

import numpy as np
import torch


def remap_labels(cache, rows):
    labels = np.asarray(cache["labels"])[rows]
    values = sorted(set(labels.tolist()))
    mapping = {value: index for index, value in enumerate(values)}
    result = {
        key: value[rows]
        for key, value in cache.items()
        if torch.is_tensor(value) and value.shape[0] == len(cache["labels"])
    }
    result["labels"] = torch.tensor([mapping[int(value)] for value in labels])
    result["domains"] = [cache["domains"][int(row)] for row in rows]
    return result, values


def label_sets(cache):
    labels = np.asarray(cache["labels"])
    domains = np.asarray(cache["domains"])
    return {
        domain: set(labels[domains == domain].tolist())
        for domain in sorted(set(cache["domains"]))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a target-specific DomainNet leave-domain-out cache."
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train_cache = torch.load(args.train_cache, weights_only=False)
    test_cache = torch.load(args.test_cache, weights_only=False)
    train_sets = label_sets(train_cache)
    test_sets = label_sets(test_cache)
    if args.target_domain not in train_sets or args.target_domain not in test_sets:
        raise ValueError(f"unknown target domain {args.target_domain}")
    source_domains = [
        domain for domain in train_sets if domain != args.target_domain
    ]
    required = [train_sets[domain] for domain in source_domains]
    required.extend([train_sets[args.target_domain], test_sets[args.target_domain]])
    allowed = set.intersection(*required)
    if not allowed:
        raise RuntimeError("empty DomainNet class intersection")

    train_rows = np.flatnonzero([
        int(value) in allowed
        for value in np.asarray(train_cache["labels"])
    ])
    test_rows = np.flatnonzero([
        value == args.target_domain
        and int(label) in allowed
        for value, label in zip(
            test_cache["domains"], np.asarray(test_cache["labels"])
        )
    ])
    train_fold, label_values = remap_labels(train_cache, train_rows)
    test_fold, test_label_values = remap_labels(test_cache, test_rows)
    if label_values != test_label_values:
        raise RuntimeError("train/test label mapping mismatch")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"target{args.target_domain}"
    train_path = os.path.join(args.out_dir, f"{stem}_train.pt")
    test_path = os.path.join(args.out_dir, f"{stem}_test.pt")
    torch.save(train_fold, train_path)
    torch.save(test_fold, test_path)
    metadata = {
        "target_domain": args.target_domain,
        "source_domains": source_domains,
        "classes": len(label_values),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_domains": {
            domain: train_fold["domains"].count(domain)
            for domain in sorted(set(train_fold["domains"]))
        },
        "test_domains": {
            domain: test_fold["domains"].count(domain)
            for domain in sorted(set(test_fold["domains"]))
        },
        "train_cache": train_path,
        "test_cache": test_path,
    }
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
