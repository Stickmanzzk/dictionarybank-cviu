import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from autoencoder.end_to_end_farg import multi_positive_contrastive
from autoencoder.rof_sae import normalised_hsic, select_code


STREAMS = ("clip", "siglip", "dino")
WIDTHS = {"clip": 768, "siglip": 768, "dino": 768}


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, strength):
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class CosineClassifier(torch.nn.Module):
    def __init__(self, input_dim, classes, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(classes, input_dim))
        self.scale = scale
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, values):
        return (
            F.normalize(values, dim=1)
            @ F.normalize(self.weight, dim=1).T
            * self.scale
        )


def make_content_classifier(input_dim, classes, kind, scale):
    if kind == "cosine":
        return CosineClassifier(input_dim, classes, scale)
    return torch.nn.Linear(input_dim, classes)


class PACSSAE(torch.nn.Module):
    def __init__(
        self,
        streams,
        widths,
        content_latents,
        nuisance_latents,
        content_k,
        nuisance_k,
        classes,
        source_domains,
        classifier_kind="linear",
        classifier_scale=12.0,
    ):
        super().__init__()
        self.streams = list(streams)
        self.content_k = content_k
        self.nuisance_k = nuisance_k
        self.pre_bias = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(widths[name]))
            for name in streams
        })
        self.content_encoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(widths[name], content_latents)
            for name in streams
        })
        self.content_decoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(content_latents, widths[name], bias=False)
            for name in streams
        })
        self.nuisance_encoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(widths[name], nuisance_latents)
            for name in streams
        })
        self.nuisance_decoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(nuisance_latents, widths[name], bias=False)
            for name in streams
        })
        self.content_classifiers = torch.nn.ModuleDict({
            name: make_content_classifier(
                content_latents,
                classes,
                classifier_kind,
                classifier_scale,
            )
            for name in streams
        })
        self.nuisance_classifiers = torch.nn.ModuleDict({
            name: torch.nn.Linear(nuisance_latents, source_domains)
            for name in streams
        })
        self.content_domain_classifier = torch.nn.Linear(
            content_latents, source_domains
        )

    def encode(self, data):
        content_logits = {
            name: self.content_encoders[name](
                data[name] - self.pre_bias[name]
            )
            for name in self.streams
        }
        content = {
            name: select_code(content_logits[name], self.content_k)
            for name in self.streams
        }
        nuisance_input = {
            name: (
                data[name]
                - self.pre_bias[name]
                - self.content_decoders[name](content[name])
            )
            for name in self.streams
        }
        nuisance_logits = {
            name: self.nuisance_encoders[name](nuisance_input[name])
            for name in self.streams
        }
        nuisance = {
            name: select_code(nuisance_logits[name], self.nuisance_k)
            for name in self.streams
        }
        return content_logits, nuisance_logits, content, nuisance

    def decode(self, content, nuisance):
        return {
            name: (
                self.pre_bias[name]
                + self.content_decoders[name](content[name])
                + self.nuisance_decoders[name](nuisance[name])
            )
            for name in self.streams
        }

    def forward(self, data):
        content_logits, nuisance_logits, content, nuisance = self.encode(data)
        reconstruction = self.decode(content, nuisance)
        return content_logits, nuisance_logits, content, nuisance, reconstruction


def split_source_rows(cache, source_domains, seed):
    domains = np.asarray(cache["domains"])
    labels = np.asarray(cache["labels"])
    fitting = []
    validation = []
    rng = np.random.default_rng(seed)
    for domain in source_domains:
        for label in sorted(set(labels.tolist())):
            rows = np.flatnonzero(
                (domains == domain) & (labels == label)
            )
            order = rng.permutation(rows)
            cut = max(1, int(len(order) * 0.8))
            fitting.extend(order[:cut].tolist())
            validation.extend(order[cut:].tolist())
    return np.asarray(fitting), np.asarray(validation)


def gather(cache, rows, streams, device):
    return {
        name: cache[name][rows].to(device)
        for name in streams
    }


def unit_norm(module):
    with torch.no_grad():
        module.weight.div_(
            module.weight.norm(dim=1, keepdim=True) + 1e-8
        )


def train(
    model,
    cache,
    fitting,
    validation,
    source_domains,
    streams,
    args,
    device,
):
    labels = torch.tensor(
        np.asarray(cache["labels"])[fitting], device=device
    )
    label_lookup = {
        int(row): int(cache["labels"][int(row)]) for row in fitting
    }
    domain_index = {
        value: index for index, value in enumerate(source_domains)
    }
    domains = torch.tensor(
        [
            domain_index[value]
            for value in np.asarray(cache["domains"])[fitting]
        ],
        device=device,
    )
    domain_lookup = {
        int(row): domain_index[cache["domains"][int(row)]]
        for row in fitting
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(fitting)
        total = 0.0
        count = 0
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            batch = gather(cache, rows, streams, device)
            (
                _,
                _,
                content,
                nuisance,
                reconstruction,
            ) = model(batch)
            batch_labels = torch.tensor(
                [label_lookup[int(row)] for row in rows],
                device=device,
            )
            batch_domains = torch.tensor(
                [domain_lookup[int(row)] for row in rows],
                device=device,
            )
            recon_loss = torch.stack([
                F.mse_loss(reconstruction[name], batch[name])
                for name in streams
            ]).mean()
            content_loss = torch.stack([
                F.cross_entropy(
                    model.content_classifiers[name](content[name]),
                    batch_labels,
                )
                for name in streams
            ]).mean()
            supervised_content = torch.stack([
                multi_positive_contrastive(
                    content[name],
                    batch_labels,
                    args.temperature,
                )
                for name in streams
            ]).mean()
            nuisance_loss = torch.stack([
                F.cross_entropy(
                    model.nuisance_classifiers[name](nuisance[name]),
                    batch_domains,
                )
                for name in streams
            ]).mean()
            mean_content = torch.stack([
                F.normalize(content[name], dim=1)
                for name in streams
            ]).mean(0)
            adversarial_logits = model.content_domain_classifier(
                GradientReverse.apply(mean_content, args.adversarial_strength)
            )
            adversarial_loss = F.cross_entropy(
                adversarial_logits, batch_domains
            )
            alignment = torch.zeros((), device=device)
            for left in range(len(streams)):
                for right in range(left + 1, len(streams)):
                    alignment = alignment + F.mse_loss(
                        content[streams[left]], content[streams[right]]
                    )
            alignment = alignment / 3.0
            contrastive = torch.stack([
                multi_positive_contrastive(
                    content[name], batch_labels, args.temperature
                )
                for name in streams
            ]).mean()
            hsic = torch.stack([
                normalised_hsic(content[name], nuisance[name])
                for name in streams
            ]).mean()
            loss = (
                recon_loss
                + args.class_weight * content_loss
                + args.supervised_content_weight * supervised_content
                + args.domain_weight * nuisance_loss
                + args.adversarial_weight * adversarial_loss
                + args.alignment_weight * alignment
                + args.contrastive_weight * contrastive
                + args.hsic_weight * hsic
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name in streams:
                for module in (
                    model.content_encoders[name],
                    model.content_decoders[name],
                    model.nuisance_encoders[name],
                    model.nuisance_decoders[name],
                ):
                    unit_norm(module)
            total += float(loss.detach()) * len(rows)
            count += len(rows)
        with torch.no_grad():
            validation_data = gather(cache, validation, streams, device)
            (
                _,
                _,
                val_content,
                val_nuisance,
                val_reconstruction,
            ) = model(validation_data)
            val_labels = torch.tensor(
                np.asarray(cache["labels"])[validation], device=device
            )
            val_domains = torch.tensor(
                [
                    domain_index[value]
                    for value in np.asarray(cache["domains"])[validation]
                ],
                device=device,
            )
            val_acc = torch.stack([
                (
                    model.content_classifiers[name](val_content[name]).argmax(1)
                    == val_labels
                ).float().mean()
                for name in streams
            ]).mean()
            val_recon = torch.stack([
                F.mse_loss(val_reconstruction[name], validation_data[name])
                for name in streams
            ]).mean()
            val_domain_acc = torch.stack([
                (
                    model.nuisance_classifiers[name](
                        val_nuisance[name]
                    ).argmax(1)
                    == val_domains
                ).float().mean()
                for name in streams
            ]).mean()
            score = float(val_acc) - 0.1 * float(val_recon) + 0.01 * float(
                val_domain_acc
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        row = {
            "epoch": epoch,
            "loss": total / max(count, 1),
            "validation_content_accuracy": float(val_acc),
            "validation_domain_accuracy": float(val_domain_acc),
            "validation_reconstruction": float(val_recon),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_target(model, cache, target_rows, streams, device):
    data = gather(cache, target_rows, streams, device)
    _, _, content, nuisance, reconstruction = model(data)
    labels = torch.tensor(
        np.asarray(cache["labels"])[target_rows], device=device
    )
    per_stream = {}
    probability_rows = []
    for name in streams:
        logits = model.content_classifiers[name](content[name])
        per_stream[name] = float((logits.argmax(1) == labels).float().mean())
        probability_rows.append(F.softmax(logits, dim=1))
    ensemble = torch.stack(probability_rows).mean(0)
    return {
        "streams": per_stream,
        "ensemble_accuracy": float(
            (ensemble.argmax(1) == labels).float().mean()
        ),
        "reconstruction_mse": float(torch.stack([
            F.mse_loss(reconstruction[name], data[name])
            for name in streams
        ]).mean()),
    }


def train_raw_baseline(
    source_cache,
    target_cache,
    fitting,
    validation,
    target_rows,
    streams,
    args,
    device,
):
    labels = np.asarray(source_cache["labels"])
    target_labels_np = np.asarray(target_cache["labels"])
    classifiers = {}
    validation_report = {}
    target_report = {}
    for name in streams:
        classifier = torch.nn.Linear(WIDTHS[name], int(labels.max()) + 1)
        classifier = classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.raw_lr)
        train_values = F.normalize(
            source_cache[name][fitting].to(device), dim=1
        )
        train_labels = torch.tensor(labels[fitting], device=device)
        for _ in range(args.raw_epochs):
            order = torch.randperm(len(train_values), device=device)
            for start in range(0, len(order), args.raw_batch_size):
                rows = order[start:start + args.raw_batch_size]
                loss = F.cross_entropy(
                    classifier(train_values[rows]), train_labels[rows]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        classifier.eval()
        classifiers[name] = classifier
        with torch.no_grad():
            val_logits = classifier(
                F.normalize(source_cache[name][validation].to(device), dim=1)
            )
            val_labels = torch.tensor(labels[validation], device=device)
            validation_report[name] = float(
                (val_logits.argmax(1) == val_labels).float().mean()
            )
            target_logits = classifier(
                F.normalize(target_cache[name][target_rows].to(device), dim=1)
            )
            target_labels = torch.tensor(
                target_labels_np[target_rows], device=device
            )
            target_report[name] = float(
                (target_logits.argmax(1) == target_labels).float().mean()
            )
    with torch.no_grad():
        validation_probabilities = []
        target_probabilities = []
        for name in streams:
            validation_probabilities.append(F.softmax(classifiers[name](
                F.normalize(source_cache[name][validation].to(device), dim=1)
            ), dim=1))
            target_probabilities.append(F.softmax(classifiers[name](
                F.normalize(target_cache[name][target_rows].to(device), dim=1)
            ), dim=1))
        validation_ensemble = torch.stack(validation_probabilities).mean(0)
        target_ensemble = torch.stack(target_probabilities).mean(0)
        validation_labels = torch.tensor(labels[validation], device=device)
        target_labels = torch.tensor(
            target_labels_np[target_rows], device=device
        )
    return {
        "streams": target_report,
        "ensemble_accuracy": float(
            (target_ensemble.argmax(1) == target_labels).float().mean()
        ),
        "validation_streams": validation_report,
        "validation_ensemble_accuracy": float(
            (validation_ensemble.argmax(1) == validation_labels).float().mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-cache", default=None)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--streams", nargs="+", default=list(STREAMS))
    parser.add_argument("--content-latents", type=int, default=2048)
    parser.add_argument("--nuisance-latents", type=int, default=128)
    parser.add_argument("--content-k", type=int, default=64)
    parser.add_argument("--nuisance-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument(
        "--classifier-kind",
        choices=("linear", "cosine"),
        default="linear",
    )
    parser.add_argument("--classifier-scale", type=float, default=12.0)
    parser.add_argument("--supervised-content-weight", type=float, default=0.0)
    parser.add_argument("--domain-weight", type=float, default=0.2)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--adversarial-strength", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--hsic-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--raw-epochs", type=int, default=100)
    parser.add_argument("--raw-batch-size", type=int, default=256)
    parser.add_argument("--raw-lr", type=float, default=1e-3)
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
    if args.target_domain not in all_domains:
        raise ValueError(f"unknown target domain {args.target_domain}")
    source_domains = [
        domain
        for domain in all_domains
        if domain != args.target_domain
        and domain not in set(args.exclude_source)
    ]
    if not source_domains:
        raise ValueError("no source domains remain after exclusions")
    args.source_domains_count = len(source_domains)
    source_rows = np.flatnonzero([
        value in source_domains for value in cache["domains"]
    ])
    source_subset = {
        **{
            name: cache[name][source_rows]
            for name in args.streams
        },
        "labels": cache["labels"][source_rows],
        "domains": [cache["domains"][int(row)] for row in source_rows],
    }
    fitting, validation = split_source_rows(source_subset, source_domains, args.seed)
    target_rows = np.flatnonzero([
        value == args.target_domain for value in target_cache["domains"]
    ])
    target_subset = {
        **{
            name: target_cache[name][target_rows]
            for name in args.streams
        },
        "labels": target_cache["labels"][target_rows],
        "domains": [args.target_domain] * len(target_rows),
    }
    target_local_rows = np.arange(len(target_rows))
    raw_report = train_raw_baseline(
        source_subset,
        target_subset,
        fitting,
        validation,
        target_local_rows,
        args.streams,
        args,
        device,
    )
    model = PACSSAE(
        args.streams,
        WIDTHS,
        args.content_latents,
        args.nuisance_latents,
        args.content_k,
        args.nuisance_k,
        int(cache["labels"].max()) + 1,
        len(source_domains),
        classifier_kind=args.classifier_kind,
        classifier_scale=args.classifier_scale,
    ).to(device)
    history = train(
        model,
        source_subset,
        fitting,
        validation,
        source_domains,
        args.streams,
        args,
        device,
    )
    model.eval()
    target_eval = evaluate_target(
        model, target_subset, target_local_rows, args.streams, device
    )
    report = {
        "settings": vars(args),
        "source_domains": source_domains,
        "target_domain": args.target_domain,
        "rows": {
            "fitting": len(fitting),
            "validation": len(validation),
            "target": len(target_rows),
        },
        "raw": raw_report,
        "factorized": target_eval,
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
