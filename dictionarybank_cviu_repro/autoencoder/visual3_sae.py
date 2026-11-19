import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from autoencoder.multistream_sae_train import MultiStreamSAE, unit_norm
from experiment_utils import half_of, metrics


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


def topk(logits, k):
    values, indices = torch.topk(logits, k, dim=1)
    codes = torch.zeros_like(logits)
    codes.scatter_(1, indices, torch.relu(values))
    return codes


def probe(rows, labels, mask):
    clf = LogisticRegression(max_iter=500)
    clf.fit(rows[mask], labels[mask])
    return float(clf.score(rows[~mask], labels[~mask]))


def build_codes(raw, values, indices, width, offset):
    local = (indices - offset).clamp(0, width - 1)
    mask = (indices >= offset) & (indices < offset + width)
    selected = torch.where(mask, torch.relu(values), torch.zeros_like(values))
    codes = torch.zeros_like(raw)
    codes.scatter_(1, local, selected)
    return codes


def scale_normalised(logits, k):
    names = list(logits)
    stacked = torch.cat([
        (logits[name] - logits[name].mean(1, keepdim=True))
        / (logits[name].std(1, keepdim=True) + 1e-6)
        for name in names
    ], dim=1)
    values, indices = torch.topk(stacked, k, dim=1)
    codes, offset = {}, 0
    for name in names:
        width = logits[name].shape[1]
        raw_values = logits[name].gather(
            1, (indices - offset).clamp(0, width - 1)
        )
        codes[name] = build_codes(
            logits[name], raw_values, indices, width, offset
        )
        offset += width
    return codes


def train(args, device):
    caches = [
        torch.load(args.train_clip, weights_only=False),
        torch.load(args.train_siglip, weights_only=False),
        torch.load(args.train_dino, weights_only=False),
    ]
    if not (
        caches[0]["contents"] == caches[1]["contents"]
        and caches[0]["contents"] == caches[2]["contents"]
    ):
        raise RuntimeError("training caches are not row-aligned")
    streams = {
        name: caches[index]["features"].to(device)
        for index, name in enumerate(WIDTHS)
    }
    model = MultiStreamSAE(WIDTHS, args.n_latents, args.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = next(iter(streams.values())).shape[0]
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = {
                name: value[order[start:start + args.batch_size]]
                for name, value in streams.items()
            }
            logits, _, recon = model(batch)
            loss = sum(
                torch.nn.functional.mse_loss(recon[name], batch[name])
                for name in batch
            )
            if args.scale_balance > 0:
                scales = torch.stack([
                    logits[name].float().std(dim=1).mean()
                    for name in model.names
                ])
                balance = (scales.log() - scales.log().mean()).pow(2).mean()
                loss = loss + args.scale_balance * balance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name in model.names:
                unit_norm(model.encoders[name])
            total += float(loss.detach()) * batch["clip"].shape[0]
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d} mse={total / n:.6f}")
    return model


def evaluate(model, args, device):
    report = {}
    for domain, files in DOMAINS.items():
        caches = [
            torch.load(os.path.join(args.cache_dir, path), weights_only=False)
            for path in files
        ]
        contents = caches[0]["contents"]
        layouts = caches[0]["layouts"]
        for cache in caches[1:]:
            if cache["contents"] != contents or cache["layouts"] != layouts:
                raise RuntimeError(f"{domain} caches are not row-aligned")
        streams = {
            name: caches[index]["features"].to(device)
            for index, name in enumerate(WIDTHS)
        }
        with torch.no_grad():
            logits, shared, _ = model(streams)
        cpu_logits = {name: value.float().cpu() for name, value in logits.items()}
        cpu_shared = {name: value.float().cpu() for name, value in shared.items()}
        labels = np.array([int(value) for value in layouts])
        mask = half_of(np.asarray(contents), seed=0)
        selected = torch.stack([
            cpu_shared[name] for name in WIDTHS
        ], dim=1)
        mass = selected.sum((0, 2))
        owners = selected.argmax(1)
        row = {
            "logit_std": {
                name: float(cpu_logits[name].std()) for name in WIDTHS
            },
            "shared_mass_share": {
                name: float(mass[index] / mass.sum().clamp_min(1e-12))
                for index, name in enumerate(WIDTHS)
            },
            "shared_ownership": {
                name: float((owners == index).float().mean())
                for index, name in enumerate(WIDTHS)
            },
            "streams": {},
        }
        normalised = scale_normalised(cpu_logits, args.k)
        for name in WIDTHS:
            released = cpu_shared[name]
            per = topk(cpu_logits[name], args.k)
            norm = normalised[name]
            row["streams"][name] = {
                "shared_slots": float((released > 0).sum(1).float().mean()),
                "shared_R@1": metrics(released, contents, layouts)["NN_R@1"],
                "per_stream_R@1": metrics(per, contents, layouts)["NN_R@1"],
                "normalised_shared_slots": float(
                    (norm > 0).sum(1).float().mean()
                ),
                "normalised_shared_R@1": metrics(
                    norm, contents, layouts
                )["NN_R@1"],
                "layout_probe": probe(released.numpy(), labels, mask),
            }
        report[domain] = row
        print(
            f"{domain:11s} "
            + " ".join(
                f"{name}:{row['shared_mass_share'][name]:.3f}/"
                f"{row['streams'][name]['shared_R@1']:.3f}/"
                f"{row['streams'][name]['per_stream_R@1']:.3f}"
                for name in WIDTHS
            )
        )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train a three-visual-stream shared top-k SAE."
    )
    parser.add_argument(
        "--train-clip", default="outputs/cache/sae_train_clip_openai.pt"
    )
    parser.add_argument(
        "--train-siglip", default="outputs/cache/sae_train_siglip.pt"
    )
    parser.add_argument(
        "--train-dino", default="outputs/cache/sae_train_dinov2_l.pt"
    )
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--n-latents", type=int, default=4096)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--scale-balance", type=float, default=0.0,
        help="Weight of the cross-stream log-scale variance penalty.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/visual3_sae.pt")
    parser.add_argument(
        "--report-json", default="outputs/visual3_sae.json"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = train(args, device)
    torch.save(
        {"state_dict": model.state_dict(), "widths": WIDTHS, "args": vars(args)},
        args.out,
    )
    report = evaluate(model, args, device)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
