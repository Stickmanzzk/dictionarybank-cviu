import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from autoencoder.cf_sae import STREAM_FILES
from autoencoder.end_to_end_farg import multi_positive_contrastive
from autoencoder.multistream_sae_train import unit_norm
from autoencoder.visual3_sae import DOMAINS, WIDTHS
from experiment_utils import half_of, metrics


class GeometryInvariantSAE(torch.nn.Module):
    def __init__(self, width, n_latents, k):
        super().__init__()
        self.pre_bias = torch.nn.Parameter(torch.zeros(width))
        self.encoder = torch.nn.Linear(width, n_latents, bias=True)
        self.k = k

    def forward(self, features):
        logits = self.encoder(features - self.pre_bias)
        values, indices = torch.topk(logits, self.k, dim=1)
        sparse = torch.zeros_like(logits).scatter_(
            1, indices, torch.relu(values)
        )
        recon = sparse @ self.encoder.weight + self.pre_bias
        return logits, sparse, recon


def train(args, features, contents, device):
    content_ids = {
        value: index for index, value in enumerate(sorted(set(contents.tolist())))
    }
    labels = torch.tensor([content_ids[value] for value in contents], device=device)
    model = GeometryInvariantSAE(
        features.shape[1], args.n_latents, args.k
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data = features.to(device)
    n = len(data)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = data[indices]
            _, sparse, recon = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            invariance = multi_positive_contrastive(
                sparse, labels[indices], args.contrastive_temperature
            )
            loss = recon_loss + args.invariance_weight * invariance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            unit_norm(model.encoder)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:3d} recon={float(recon_loss.detach()):.5f} "
                f"inv={float(invariance.detach()):.5f}"
            )
    return model


def probe(rows, labels, mask):
    classifier = LogisticRegression(max_iter=700)
    classifier.fit(rows[mask], labels[mask])
    return float(classifier.score(rows[~mask], labels[~mask]))


def evaluate(model, args, device):
    if args.test_cache:
        cache = torch.load(args.test_cache, weights_only=False)
        features = cache[args.stream].to(device)
        with torch.no_grad():
            _, sparse, recon = model(features)
        contents = cache["contents"]
        layouts = cache["layouts"]
        labels = np.asarray([int(value) for value in layouts])
        mask = half_of(np.asarray(contents), seed=0)
        row = {
            "content_R@1": metrics(
                sparse.float().cpu(), contents, layouts
            )["NN_R@1"],
            "slots": float((sparse > 0).sum(1).float().mean()),
            "layout_probe": probe(
                sparse.float().cpu().numpy(), labels, mask
            ),
            "recon_mse": float(
                torch.nn.functional.mse_loss(recon, features)
            ),
        }
        print("cifar100", row)
        return {"cifar100": row}
    report = {}
    for domain, patterns in DOMAINS.items():
        path = os.path.join(
            args.cache_dir,
            patterns[list(WIDTHS).index(args.stream)].format(
                stream=STREAM_FILES[args.stream]
            ) if "{stream}" in patterns[0] else patterns[
                list(WIDTHS).index(args.stream)
            ],
        )
        cache = torch.load(path, weights_only=False)
        contents = cache["contents"]
        layouts = cache["layouts"]
        features = cache["features"].to(device)
        with torch.no_grad():
            _, sparse, recon = model(features)
        layout_labels = np.asarray([int(value) for value in layouts])
        mask = half_of(np.asarray(contents), seed=0)
        report[domain] = {
            "content_R@1": metrics(
                sparse.float().cpu(), contents, layouts
            )["NN_R@1"],
            "slots": float((sparse > 0).sum(1).float().mean()),
            "layout_probe": probe(
                sparse.float().cpu().numpy(), layout_labels, mask
            ),
            "recon_mse": float(
                torch.nn.functional.mse_loss(recon, features)
            ),
        }
        print(domain, report[domain])
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Geometry-invariant SAE baseline."
    )
    parser.add_argument("--stream", choices=tuple(WIDTHS), default="clip")
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--test-cache", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--n-latents", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--invariance-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    args.out = args.out or f"outputs/gi_sae_{args.stream}.pt"
    args.report_json = args.report_json or f"outputs/gi_sae_{args.stream}.json"

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.train_cache:
        cache = torch.load(args.train_cache, weights_only=False)
        if args.stream not in cache:
            raise KeyError(f"{args.stream} is missing from {args.train_cache}")
        features = cache[args.stream]
    else:
        cache = torch.load(
            os.path.join(
                args.cache_dir, f"sae_train_{STREAM_FILES[args.stream]}.pt"
            ),
            weights_only=False,
        )
        features = cache["features"]
    model = train(
        args,
        features,
        np.asarray(cache["contents"]),
        device,
    )
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out)
    report = evaluate(model, args, device)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
