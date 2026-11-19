import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from autoencoder.pacs_factor_sae import (
    PACSSAE,
    WIDTHS,
    split_source_rows,
)


def normalise(values):
    return F.normalize(values, dim=1)


def load_factor_model(checkpoint, cache, streams, device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = blob["args"]
    model = PACSSAE(
        streams,
        WIDTHS,
        saved["content_latents"],
        saved["nuisance_latents"],
        saved["content_k"],
        saved["nuisance_k"],
        int(cache["labels"].max()) + 1,
        len(set(cache["domains"])) - 1,
        classifier_kind=saved.get("classifier_kind", "linear"),
        classifier_scale=saved.get("classifier_scale", 12.0),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved


@torch.no_grad()
def content_codes(model, data):
    _, _, content, _ = model.encode(data)
    return content


def train_stream(
    args,
    raw_train,
    code_train,
    labels,
    raw_validation,
    code_validation,
    validation_labels,
    raw_target,
    code_target,
    device,
):
    head = torch.nn.Linear(code_train.shape[1], raw_train.shape[1], bias=False)
    head = head.to(device)
    torch.nn.init.zeros_(head.weight)
    classifier = torch.nn.Linear(raw_train.shape[1], labels.max().item() + 1)
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(
        list(head.parameters()) + list(classifier.parameters()), lr=args.lr
    )
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(raw_train), device=device)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            representation = normalise(
                raw_train[rows]
                + args.alpha * head(code_train[rows])
            )
            loss = F.cross_entropy(classifier(representation), labels[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        with torch.no_grad():
            validation_representation = normalise(
                raw_validation + args.alpha * head(code_validation)
            )
            accuracy = float(
                (
                    classifier(validation_representation).argmax(1)
                    == validation_labels
                ).float().mean()
            )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                "head": {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                },
                "classifier": {
                    key: value.detach().cpu().clone()
                    for key, value in classifier.state_dict().items()
                },
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d} loss={total / len(raw_train):.5f} "
                f"validation={accuracy:.4f}"
            )
    head.load_state_dict(best_state["head"])
    classifier.load_state_dict(best_state["classifier"])
    head.eval()
    classifier.eval()
    with torch.no_grad():
        representation = normalise(
            raw_target + args.alpha * head(code_target)
        )
        logits = classifier(representation)
    return {
        "validation_accuracy": best_accuracy,
        "target_logits": logits.cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", default=None)
    parser.add_argument("--factor-checkpoint", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--streams", nargs="+", default=["clip", "siglip"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache = torch.load(args.cache, weights_only=False)
    target_cache = torch.load(
        args.target_cache or args.cache, weights_only=False
    )
    all_domains = sorted(set(cache["domains"]))
    source_domains = [
        domain for domain in all_domains if domain != args.target_domain
    ]
    source_rows = np.flatnonzero([
        value in source_domains for value in cache["domains"]
    ])
    source_cache = {
        **{name: cache[name][source_rows] for name in args.streams},
        "labels": cache["labels"][source_rows],
        "domains": [cache["domains"][int(row)] for row in source_rows],
    }
    fitting, validation = split_source_rows(
        source_cache, source_domains, args.seed
    )
    target_rows = np.flatnonzero([
        value == args.target_domain for value in target_cache["domains"]
    ])
    model, saved = load_factor_model(
        args.factor_checkpoint, cache, args.streams, device
    )
    source_data = {
        name: source_cache[name].to(device) for name in args.streams
    }
    source_codes = content_codes(model, source_data)
    target_data = {
        name: target_cache[name][target_rows].to(device)
        for name in args.streams
    }
    target_codes = content_codes(model, target_data)
    fitting_tensor = torch.tensor(fitting, device=device)
    validation_tensor = torch.tensor(validation, device=device)
    labels = torch.tensor(
        np.asarray(source_cache["labels"])[fitting], device=device
    )
    validation_labels = torch.tensor(
        np.asarray(source_cache["labels"])[validation], device=device
    )
    target_labels = torch.tensor(
        np.asarray(target_cache["labels"])[target_rows], device=device
    )
    report = {
        "settings": vars(args),
        "factor_settings": saved,
        "source_domains": source_domains,
        "target_domain": args.target_domain,
        "streams": {},
    }
    probabilities = []
    for name in args.streams:
        result = train_stream(
            args,
            source_data[name][fitting_tensor],
            source_codes[name][fitting_tensor],
            labels,
            source_data[name][validation_tensor],
            source_codes[name][validation_tensor],
            validation_labels,
            target_data[name],
            target_codes[name],
            device,
        )
        target_logits = result.pop("target_logits").to(device)
        probabilities.append(F.softmax(target_logits, dim=1))
        report["streams"][name] = {
            **result,
            "target_accuracy": float(
                (target_logits.argmax(1) == target_labels).float().mean()
            ),
        }
    ensemble = torch.stack(probabilities).mean(0)
    report["ensemble_accuracy"] = float(
        (ensemble.argmax(1) == target_labels).float().mean()
    )
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
