import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch

from autoencoder.multistream_sae_train import MultiStreamSAE, unit_norm
from autoencoder.visual3_sae import WIDTHS, evaluate


class FARGSAE(MultiStreamSAE):
    def __init__(self, widths, n_latents, k, utility_strength):
        super().__init__(widths, n_latents, k)
        self.utility_strength = utility_strength
        self.utility = {
            name: torch.zeros(n_latents) for name in self.names
        }

    def set_utility(self, utility):
        self.utility = {
            name: utility[name].detach().to(self.encoders[name].weight.device)
            for name in self.names
        }

    def forward(self, streams):
        logits = {
            name: self.encoders[name](streams[name] - self.pre_bias[name])
            for name in self.names
        }
        scores = []
        for name in self.names:
            values = logits[name]
            ranks = values.argsort(dim=1).argsort(dim=1).float()
            ranks = ranks / max(1, values.shape[1] - 1)
            scores.append(
                ranks
                + self.utility_strength
                * self.utility[name].to(values.device)[None, :]
            )
        stacked = torch.cat(scores, dim=1)
        _, indices = torch.topk(stacked, self.k, dim=1)
        sparse = {name: torch.zeros_like(logits[name]) for name in self.names}
        offset = 0
        for name in self.names:
            mask = (indices >= offset) & (indices < offset + self.n_latents)
            local = (indices - offset).clamp(0, self.n_latents - 1)
            values = logits[name].gather(1, local)
            selected = torch.where(
                mask, torch.relu(values), torch.zeros_like(values)
            )
            sparse[name].scatter_(1, local, selected)
            offset += self.n_latents
        recon = {
            name: sparse[name] @ self.encoders[name].weight + self.pre_bias[name]
            for name in self.names
        }
        return logits, sparse, recon


def within_group_variance(values, groups, n_groups):
    counts = torch.bincount(groups, minlength=n_groups).float()
    sums = torch.zeros(
        n_groups, values.shape[1], device=values.device
    )
    sums.index_add_(0, groups, values.float())
    means = sums / counts[:, None].clamp_min(1)
    centred = values.float() - means[groups]
    squares = torch.zeros_like(sums)
    squares.index_add_(0, groups, centred.pow(2))
    per_group = squares / counts[:, None].clamp_min(1)
    return per_group.mean(0)


@torch.no_grad()
def estimate_utility(model, streams, contents, layouts, batch_size):
    content_values = sorted(set(contents.tolist()))
    layout_values = sorted(set(layouts.tolist()))
    content_index = {value: index for index, value in enumerate(content_values)}
    layout_index = {value: index for index, value in enumerate(layout_values)}
    content_groups = torch.tensor(
        [content_index[value] for value in contents],
        device=next(iter(streams.values())).device,
    )
    layout_groups = torch.tensor(
        [layout_index[value] for value in layouts],
        device=next(iter(streams.values())).device,
    )
    logits = {name: [] for name in WIDTHS}
    n = next(iter(streams.values())).shape[0]
    for start in range(0, n, batch_size):
        batch = {
            name: value[start:start + batch_size]
            for name, value in streams.items()
        }
        current, _, _ = model(batch)
        for name in WIDTHS:
            logits[name].append(current[name].float().cpu())
    utility = {}
    for name in WIDTHS:
        values = torch.cat(logits[name]).to(content_groups.device)
        nuisance_var = within_group_variance(
            values, content_groups, len(content_values)
        )
        content_var = within_group_variance(
            values, layout_groups, len(layout_values)
        )
        values = torch.log(
            (content_var + 1e-6) / (nuisance_var + 1e-6)
        )
        utility[name] = (
            (values - values.mean()) / (values.std() + 1e-6)
        ).cpu()
    return utility


def raw_gate_recon(model, logits):
    stacked = torch.cat([logits[name] for name in model.names], dim=1)
    _, indices = torch.topk(stacked, model.k, dim=1)
    codes, offset = {}, 0
    for name in model.names:
        mask = (indices >= offset) & (indices < offset + model.n_latents)
        local = (indices - offset).clamp(0, model.n_latents - 1)
        values = logits[name].gather(1, local)
        selected = torch.where(
            mask, torch.relu(values), torch.zeros_like(values)
        )
        code = torch.zeros_like(logits[name])
        code.scatter_(1, local, selected)
        codes[name] = code
        offset += model.n_latents
    return {
        name: codes[name] @ model.encoders[name].weight + model.pre_bias[name]
        for name in model.names
    }


def multi_positive_contrastive(codes, labels, temperature):
    normalised = codes / (codes.norm(dim=1, keepdim=True) + 1e-8)
    similarity = normalised @ normalised.T / temperature
    positive = labels[:, None].eq(labels[None, :])
    positive.fill_diagonal_(False)
    valid = positive.any(dim=1)
    if not valid.any():
        return codes.new_zeros(())
    log_prob = torch.log_softmax(similarity, dim=1)
    positive_log_prob = log_prob.masked_fill(~positive, float("-inf"))
    per_row = torch.logsumexp(positive_log_prob, dim=1)
    per_row = per_row - positive.sum(1).clamp_min(1).float().log()
    return -per_row[valid].mean()


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
    contents = np.asarray(caches[0]["contents"])
    layouts = np.asarray(caches[0]["layouts"])
    content_ids = {
        value: index for index, value in enumerate(sorted(set(contents.tolist())))
    }
    content_tensor = torch.tensor(
        [content_ids[value] for value in contents], device=device
    )
    streams = {
        name: caches[index]["features"].to(device)
        for index, name in enumerate(WIDTHS)
    }
    model = FARGSAE(
        WIDTHS, args.n_latents, args.k, args.utility_strength
    ).to(device)
    if args.init_checkpoint:
        blob = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        print(f"initialized from {args.init_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = next(iter(streams.values())).shape[0]
    for epoch in range(1, args.epochs + 1):
        if (
            epoch == 1
            or (
                not args.freeze_utility
                and epoch % args.utility_every == 0
            )
        ):
            utility = estimate_utility(
                model, streams, contents, layouts, args.batch_size
            )
            model.set_utility(utility)
            print(
                "epoch",
                epoch,
                "utility std",
                {name: float(value.std()) for name, value in utility.items()},
            )
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = {
                name: value[order[start:start + args.batch_size]]
                for name, value in streams.items()
            }
            logits, sparse, farg_recon = model(batch)
            farg_loss = sum(
                torch.nn.functional.mse_loss(farg_recon[name], batch[name])
                for name in batch
            )
            raw_recon = raw_gate_recon(model, logits)
            raw_loss = sum(
                torch.nn.functional.mse_loss(raw_recon[name], batch[name])
                for name in batch
            )
            contrastive = torch.stack([
                multi_positive_contrastive(
                    sparse[name], content_tensor[order[start:start + args.batch_size]],
                    args.contrastive_temperature,
                )
                for name in model.names
            ]).mean()
            loss = (
                args.raw_weight * raw_loss
                + args.farg_weight * farg_loss
                + args.content_weight * contrastive
            )
            if args.gate_balance > 0:
                slots = torch.stack([
                    (sparse[name] > 0).float().sum(1) / args.k
                    for name in model.names
                ])
                target = torch.full_like(slots, 1.0 / len(model.names))
                loss = loss + args.gate_balance * (slots - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name in model.names:
                unit_norm(model.encoders[name])
            total += float(loss.detach()) * batch["clip"].shape[0]
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:3d} total={total / n:.6f} "
                f"raw={float(raw_loss.detach()):.6f} "
                f"farg={float(farg_loss.detach()):.6f} "
                f"content={float(contrastive.detach()):.6f}"
            )
    return model, streams, contents, layouts


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training with a factor-aware rank gate."
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--utility-strength", type=float, default=0.1)
    parser.add_argument("--utility-every", type=int, default=10)
    parser.add_argument("--freeze-utility", action="store_true")
    parser.add_argument("--gate-balance", type=float, default=0.0)
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--farg-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--out", default="outputs/farg_sae.pt")
    parser.add_argument("--report-json", default="outputs/farg_sae.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, streams, contents, layouts = train(args, device)
    final_utility = estimate_utility(
        model, streams, contents, layouts, args.batch_size
    )
    model.set_utility(final_utility)
    torch.save({
        "state_dict": model.state_dict(),
        "utility": final_utility,
        "args": vars(args),
    }, args.out)
    eval_args = argparse.Namespace(cache_dir=args.cache_dir, k=args.k)
    report = evaluate(model, eval_args, device)
    for domain, row in report.items():
        row["farg_strength"] = args.utility_strength
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
