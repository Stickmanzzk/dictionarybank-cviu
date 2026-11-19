import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from experiment_utils import half_of, metrics


DOMAIN_CACHES = {
    "documents": "documents_{stream}.pt",
    "shapes": "shapes_{stream}.pt",
    "real scans": "real_scans_funsd_{stream}.pt",
    "arxiv": "real_scans_wide_{stream}.pt",
}
STREAM_FILES = {
    "clip": "clip_openai",
    "siglip": "siglip",
    "dino": "dinov2_l",
}
STREAM_KEYS = {
    "clip": "clip",
    "siglip": "siglip",
    "dino": "dino",
}
WIDTHS = {"clip": 768, "siglip": 768, "dino": 1024}


class CounterfactualSAE(torch.nn.Module):
    def __init__(self, width, content_latents, nuisance_latents,
                 content_k, nuisance_k):
        super().__init__()
        self.pre_bias = torch.nn.Parameter(torch.zeros(width))
        self.content_encoder = torch.nn.Linear(width, content_latents)
        self.nuisance_encoder = torch.nn.Linear(width, nuisance_latents)
        self.decoder = torch.nn.Linear(
            content_latents + nuisance_latents, width
        )
        self.content_k = content_k
        self.nuisance_k = nuisance_k

    def encode(self, features):
        centred = features - self.pre_bias
        content_logits = self.content_encoder(centred)
        nuisance_logits = self.nuisance_encoder(centred)
        content_values, content_indices = torch.topk(
            content_logits, self.content_k, dim=1
        )
        nuisance_values, nuisance_indices = torch.topk(
            nuisance_logits, self.nuisance_k, dim=1
        )
        content = torch.zeros_like(content_logits).scatter_(
            1, content_indices, torch.relu(content_values)
        )
        nuisance = torch.zeros_like(nuisance_logits).scatter_(
            1, nuisance_indices, torch.relu(nuisance_values)
        )
        return content, nuisance

    def decode(self, content, nuisance):
        return self.decoder(torch.cat([content, nuisance], dim=1)) + self.pre_bias

    def forward(self, features):
        content, nuisance = self.encode(features)
        recon = self.decode(content, nuisance)
        return content, nuisance, recon


def unit_norm(module):
    with torch.no_grad():
        module.weight.div_(
            module.weight.norm(dim=1, keepdim=True) + 1e-8
        )


def build_lookup(contents, layouts):
    content_values = sorted(set(contents.tolist()))
    layout_values = sorted(set(layouts.tolist()))
    content_index = {value: index for index, value in enumerate(content_values)}
    layout_index = {value: index for index, value in enumerate(layout_values)}
    lookup = torch.full(
        (len(content_values), len(layout_values)), -1, dtype=torch.long
    )
    for index, (content, layout) in enumerate(zip(contents, layouts)):
        lookup[content_index[content], layout_index[layout]] = index
    if (lookup < 0).any():
        raise RuntimeError("the training grid is not a complete content-by-nuisance grid")
    return lookup, content_index, layout_index


def random_triplets(lookup, batch_size, generator):
    n_contents, n_nuisances = lookup.shape
    anchor_content = torch.randint(
        0, n_contents, (batch_size,), generator=generator
    )
    anchor_layout = torch.randint(
        0, n_nuisances, (batch_size,), generator=generator
    )
    target_layout = torch.randint(
        0, n_nuisances, (batch_size,), generator=generator
    )
    same = target_layout == anchor_layout
    target_layout[same] = (target_layout[same] + 1) % n_nuisances
    donor_content = torch.randint(
        0, n_contents, (batch_size,), generator=generator
    )
    same = donor_content == anchor_content
    donor_content[same] = (donor_content[same] + 1) % n_contents
    anchor = lookup[anchor_content, anchor_layout]
    target = lookup[anchor_content, target_layout]
    donor = lookup[donor_content, target_layout]
    return anchor, target, donor


def train(args, features, contents, layouts, device):
    lookup, _, _ = build_lookup(contents, layouts)
    generator = torch.Generator().manual_seed(args.seed)
    model = CounterfactualSAE(
        features.shape[1],
        args.content_latents,
        args.nuisance_latents,
        args.content_k,
        args.nuisance_k,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data = features.to(device)
    n = len(data)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            normal = order[start:start + args.batch_size]
            anchor, target, donor = random_triplets(
                lookup,
                len(normal),
                torch.Generator().manual_seed(
                    args.seed * 100000 + epoch * 1000 + start
                ),
            )
            anchor = anchor.to(device)
            target = target.to(device)
            donor = donor.to(device)
            content_a, nuisance_a = model.encode(data[normal])
            recon_a = model.decode(content_a, nuisance_a)
            content_t, nuisance_t = model.encode(data[target])
            content_d, nuisance_d = model.encode(data[donor])
            normal_loss = torch.nn.functional.mse_loss(recon_a, data[normal])
            invariance = torch.nn.functional.mse_loss(content_a, content_t)
            swapped_t = model.decode(content_a, nuisance_d)
            swap_loss = torch.nn.functional.mse_loss(
                swapped_t, data[target]
            )
            swapped_a = model.decode(content_t, nuisance_a)
            swap_loss = swap_loss + torch.nn.functional.mse_loss(
                swapped_a, data[anchor]
            )
            nuisance_diversity = -torch.nn.functional.mse_loss(
                nuisance_a, nuisance_d
            )
            loss = (
                normal_loss
                + args.invariance_weight * invariance
                + args.swap_weight * swap_loss
                + args.diversity_weight * nuisance_diversity
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            unit_norm(model.content_encoder)
            unit_norm(model.nuisance_encoder)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:3d} recon={float(normal_loss.detach()):.5f} "
                f"inv={float(invariance.detach()):.5f} "
                f"swap={float(swap_loss.detach()):.5f}"
            )
    return model


def probe(rows, labels, mask):
    classifier = LogisticRegression(max_iter=700)
    classifier.fit(rows[mask], labels[mask])
    return float(classifier.score(rows[~mask], labels[~mask]))


def evaluate(model, args, device):
    report = {}
    for domain, pattern in DOMAIN_CACHES.items():
        path = os.path.join(
            args.cache_dir,
            pattern.format(stream=STREAM_FILES[args.stream]),
        )
        cache = torch.load(path, weights_only=False)
        contents = cache["contents"]
        layouts = cache["layouts"]
        features = cache["features"].to(device)
        with torch.no_grad():
            content, nuisance, recon = model(features)
        labels = np.asarray([int(value) for value in layouts])
        mask = half_of(np.asarray(contents), seed=0)
        report[domain] = {
            "recon_mse": float(
                torch.nn.functional.mse_loss(recon, features)
            ),
            "content_R@1": metrics(
                content.cpu(), contents, layouts
            )["NN_R@1"],
            "content_layout_probe": probe(
                content.cpu().numpy(), labels, mask
            ),
            "nuisance_content_probe": 0.0,
            "nuisance_layout_probe": probe(
                nuisance.cpu().numpy(), labels, mask
            ),
        }
        print(domain, report[domain])
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual sparse autoencoder on paired views."
    )
    parser.add_argument("--stream", choices=tuple(WIDTHS), default="clip")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--content-latents", type=int, default=4096)
    parser.add_argument("--nuisance-latents", type=int, default=1024)
    parser.add_argument("--content-k", type=int, default=32)
    parser.add_argument("--nuisance-k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    args.out = args.out or f"outputs/cf_sae_{args.stream}.pt"
    args.report_json = (
        args.report_json or f"outputs/cf_sae_{args.stream}.json"
    )

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_path = os.path.join(
        args.cache_dir, f"sae_train_{STREAM_FILES[args.stream]}.pt"
    )
    train_cache = torch.load(train_path, weights_only=False)
    model = train(
        args,
        train_cache["features"],
        np.asarray(train_cache["contents"]),
        np.asarray(train_cache["layouts"]),
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
