import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from autoencoder.cf_sae import build_lookup, random_triplets
from autoencoder.end_to_end_farg import multi_positive_contrastive
from autoencoder.gate_variants import three_way_split
from experiment_utils import half_of, metrics
from factor_eval import variance_decomposition


WIDTHS = {"clip": 768, "siglip": 768, "dino": 768}


def select_code(logits, k, block_size=0):
    if block_size <= 1:
        values, indices = torch.topk(logits, k, dim=1)
        return torch.zeros_like(logits).scatter_(
            1, indices, F.relu(values)
        )

    width = logits.shape[1]
    groups = (width + block_size - 1) // block_size
    padded_width = groups * block_size
    padded = logits.new_full((logits.shape[0], padded_width), -torch.inf)
    padded[:, :width] = logits
    grouped = padded.reshape(logits.shape[0], groups, block_size)
    keep_groups = min(groups, (k + block_size - 1) // block_size)
    group_scores = grouped.max(dim=2).values
    _, kept = torch.topk(group_scores, keep_groups, dim=1)
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, kept, True)
    candidate = padded.masked_fill(
        ~group_mask.repeat_interleave(block_size, dim=1), -torch.inf
    )
    values, indices = torch.topk(candidate[:, :width], k, dim=1)
    return torch.zeros_like(logits).scatter_(
        1, indices, F.relu(logits.gather(1, indices))
    )


def normalised_hsic(left, right):
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    gram_left = left @ left.transpose(0, 1)
    gram_right = right @ right.transpose(0, 1)
    gram_left = gram_left - gram_left.mean(dim=0, keepdim=True)
    gram_left = gram_left - gram_left.mean(dim=1, keepdim=True)
    gram_right = gram_right - gram_right.mean(dim=0, keepdim=True)
    gram_right = gram_right - gram_right.mean(dim=1, keepdim=True)
    numerator = (gram_left * gram_right).sum()
    denominator = gram_left.square().sum().sqrt()
    denominator = denominator * gram_right.square().sum().sqrt() + 1e-8
    return numerator / denominator


def paired_contrastive(left, right, labels, temperature):
    left = left / (left.norm(dim=1, keepdim=True) + 1e-8)
    right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    similarity = left @ right.transpose(0, 1) / temperature
    positive = labels[:, None].eq(labels[None, :])
    valid = positive.any(dim=1)

    def direction(logits):
        log_prob = torch.log_softmax(logits, dim=1)
        positive_log_prob = log_prob.masked_fill(
            ~positive, float("-inf")
        )
        value = torch.logsumexp(positive_log_prob, dim=1)
        value = value - positive.sum(1).clamp_min(1).float().log()
        return -value[valid].mean()

    return 0.5 * (direction(similarity) + direction(similarity.transpose(0, 1)))


class ResidualOrthogonalFactorizedSAE(torch.nn.Module):
    def __init__(
        self,
        streams,
        widths,
        content_latents,
        nuisance_latents,
        content_k,
        nuisance_k,
        residual_nuisance=True,
        nested_k=None,
        block_size=0,
        retrieval_dim=0,
        retrieval_anchor=False,
        retrieval_alpha=0.1,
        retrieval_from_raw=False,
        retrieval_regularization=1e-4,
        retrieval_hidden_dim=0,
    ):
        super().__init__()
        self.streams = list(streams)
        self.content_k = content_k
        self.nuisance_k = nuisance_k
        self.residual_nuisance = residual_nuisance
        self.nested_k = tuple(sorted(set(nested_k or (content_k,))))
        self.block_size = block_size
        self.retrieval_dim = retrieval_dim
        self.retrieval_anchor = retrieval_anchor
        self.retrieval_alpha = retrieval_alpha
        self.retrieval_from_raw = retrieval_from_raw
        self.retrieval_regularization = retrieval_regularization
        self.retrieval_hidden_dim = retrieval_hidden_dim
        self.content_encoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(widths[name], content_latents)
            for name in self.streams
        })
        self.content_decoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(content_latents, widths[name], bias=False)
            for name in self.streams
        })
        self.nuisance_encoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(widths[name], nuisance_latents)
            for name in self.streams
        })
        self.nuisance_decoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(nuisance_latents, widths[name], bias=False)
            for name in self.streams
        })
        self.pre_bias = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(widths[name]))
            for name in self.streams
        })
        if retrieval_from_raw:
            self.retrieval_heads = torch.nn.ModuleDict({
                name: (
                    torch.nn.Sequential(
                        torch.nn.Linear(widths[name], retrieval_hidden_dim),
                        torch.nn.GELU(),
                        torch.nn.Linear(
                            retrieval_hidden_dim, widths[name], bias=False
                        ),
                    )
                    if retrieval_hidden_dim > 0
                    else torch.nn.Linear(
                        widths[name], widths[name], bias=False
                    )
                )
                for name in self.streams
            })
            for head in self.retrieval_heads.values():
                final = (
                    head[-1] if isinstance(head, torch.nn.Sequential) else head
                )
                torch.nn.init.zeros_(final.weight)
        elif retrieval_dim > 0:
            self.retrieval_heads = torch.nn.ModuleDict({
                name: torch.nn.Linear(
                    content_latents,
                    widths[name] if retrieval_anchor else retrieval_dim,
                    bias=False,
                )
                for name in self.streams
            })
            for head in self.retrieval_heads.values():
                torch.nn.init.zeros_(head.weight)

    def retrieval_code(self, content, streams=None):
        if not hasattr(self, "retrieval_heads"):
            return content
        if self.retrieval_from_raw:
            if streams is None:
                raise ValueError("raw retrieval adapter requires input streams")
            return {
                name: F.normalize(
                    streams[name]
                    + self.retrieval_alpha
                    * self.retrieval_heads[name](streams[name]),
                    dim=1,
                )
                for name in self.streams
            }
        if self.retrieval_anchor and streams is None:
            raise ValueError("anchored retrieval requires the input streams")
        return {
            name: F.normalize(
                streams[name]
                + self.retrieval_alpha * self.retrieval_heads[name](content[name])
                if self.retrieval_anchor
                else self.retrieval_heads[name](content[name]),
                dim=1,
            )
            for name in self.streams
        }

    def encode_content(self, streams):
        logits = {
            name: self.content_encoders[name](streams[name] - self.pre_bias[name])
            for name in self.streams
        }
        codes = {
            name: select_code(
                logits[name], self.content_k, self.block_size
            )
            for name in self.streams
        }
        return logits, codes

    def encode(self, streams):
        content_logits, content = self.encode_content(streams)
        nuisance_input = {}
        for name in self.streams:
            content_recon = self.content_decoders[name](content[name])
            nuisance_input[name] = (
                streams[name] - self.pre_bias[name] - content_recon
                if self.residual_nuisance
                else streams[name] - self.pre_bias[name]
            )
        nuisance_logits = {
            name: self.nuisance_encoders[name](nuisance_input[name])
            for name in self.streams
        }
        nuisance = {
            name: select_code(
                nuisance_logits[name], self.nuisance_k, self.block_size
            )
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

    def nested_content(self, logits):
        return [
            {
                name: select_code(
                    logits[name], budget, self.block_size
                )
                for name in self.streams
            }
            for budget in self.nested_k
        ]

    def forward(self, streams):
        content_logits, nuisance_logits, content, nuisance = self.encode(streams)
        return content_logits, nuisance_logits, content, nuisance


def unit_norm(module):
    with torch.no_grad():
        module.weight.div_(
            module.weight.norm(dim=1, keepdim=True) + 1e-8
        )


def gather(data, indices, names):
    return {name: data[name][indices] for name in names}


def train(args, data, contents, layouts, device, triplet_sampler=None):
    seed = torch.Generator().manual_seed(args.seed)
    lookup = None
    if triplet_sampler is None:
        lookup, _, _ = build_lookup(contents, layouts)
    content_ids = {
        value: index for index, value in enumerate(sorted(set(contents.tolist())))
    }
    content_labels = torch.tensor(
        [content_ids[value] for value in contents], device=device
    )
    model = ResidualOrthogonalFactorizedSAE(
        args.streams,
        {name: data[name].shape[1] for name in args.streams},
        args.content_latents,
        args.nuisance_latents,
        args.content_k,
        args.nuisance_k,
        residual_nuisance=not args.joint_nuisance,
        nested_k=args.nested_k,
        block_size=args.block_size,
        retrieval_dim=getattr(args, "retrieval_dim", 0),
        retrieval_anchor=getattr(args, "retrieval_anchor", False),
        retrieval_alpha=getattr(args, "retrieval_alpha", 0.1),
        retrieval_from_raw=getattr(args, "retrieval_from_raw", False),
        retrieval_regularization=getattr(
            args, "retrieval_regularization", 1e-4
        ),
        retrieval_hidden_dim=getattr(args, "retrieval_hidden_dim", 0),
    ).to(device)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint:
        blob = torch.load(init_checkpoint, map_location=device, weights_only=False)
        current = model.state_dict()
        compatible = {
            name: value
            for name, value in blob["state_dict"].items()
            if name in current and current[name].shape == value.shape
        }
        incompatible = model.load_state_dict(compatible, strict=False)
        print(
            f"initialized from {init_checkpoint}; "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    if getattr(args, "freeze_core", False):
        for name, parameter in model.named_parameters():
            if not name.startswith("retrieval_heads."):
                parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = next(iter(data.values())).shape[0]
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        totals = {
            "loss": 0.0,
            "recon": 0.0,
            "invariance": 0.0,
            "alignment": 0.0,
            "swap": 0.0,
            "hsic": 0.0,
            "contrastive": 0.0,
            "cross_modal": 0.0,
        }
        for start in range(0, n, args.batch_size):
            normal = order[start:start + args.batch_size]
            if triplet_sampler is None:
                anchor, target, donor = random_triplets(
                    lookup,
                    len(normal),
                    torch.Generator().manual_seed(
                        args.seed * 100000 + epoch * 1000 + start
                    ),
                )
            else:
                anchor, target, donor = triplet_sampler.sample(
                    len(normal),
                    args.seed * 100000 + epoch * 1000 + start,
                )
            anchor = anchor.to(device)
            target = target.to(device)
            donor = donor.to(device)
            normal_batch = gather(data, normal, args.streams)
            anchor_batch = gather(data, anchor, args.streams)
            target_batch = gather(data, target, args.streams)
            donor_batch = gather(data, donor, args.streams)

            content_logits, _, content_n, nuisance_n = model(normal_batch)
            _, _, content_a, nuisance_a = model(anchor_batch)
            _, _, _, nuisance_d = model(donor_batch)
            recon_n = model.decode(content_n, nuisance_n)

            nested_loss = torch.zeros((), device=device)
            nested_codes = model.nested_content(content_logits)
            for budget, codes in zip(model.nested_k, nested_codes):
                nested_recon = {
                    name: model.pre_bias[name] + model.content_decoders[name](
                        codes[name]
                    )
                    for name in args.streams
                }
                nested_loss = nested_loss + sum(
                    F.mse_loss(nested_recon[name], normal_batch[name])
                    for name in args.streams
                ) / len(model.nested_k)
            recon_loss = (
                sum(
                    F.mse_loss(recon_n[name], normal_batch[name])
                    for name in args.streams
                )
                + args.matryoshka_weight * nested_loss
            )
            invariance = sum(
                F.mse_loss(content_a[name], content_n[name])
                for name in args.streams
            ) / len(args.streams)
            alignment = torch.zeros((), device=device)
            for left in range(len(args.streams)):
                for right in range(left + 1, len(args.streams)):
                    alignment = alignment + F.mse_loss(
                        content_a[args.streams[left]],
                        content_a[args.streams[right]],
                    )
            swap_loss = torch.zeros((), device=device)
            for name in args.streams:
                swapped = model.decode(content_a, nuisance_d)[name]
                swap_loss = swap_loss + F.mse_loss(
                    swapped, target_batch[name]
                )
                swapped = model.decode(content_n, nuisance_a)[name]
                swap_loss = swap_loss + F.mse_loss(
                    swapped, anchor_batch[name]
                )
            swap_loss = swap_loss / (2 * len(args.streams))
            hsic = torch.stack([
                normalised_hsic(content_n[name], nuisance_n[name])
                for name in args.streams
            ]).mean()
            contrastive = torch.stack([
                multi_positive_contrastive(
                    content_n[name],
                    content_labels[normal],
                    args.contrastive_temperature,
                )
                for name in args.streams
            ]).mean()
            cross_modal = torch.zeros((), device=device)
            cross_modal_weight = getattr(args, "cross_modal_weight", 0.0)
            if cross_modal_weight > 0:
                pairs = (
                    ("clip_img", "clip_txt"),
                    ("siglip_img", "siglip_txt"),
                )
                projected = model.retrieval_code(content_n, normal_batch)
                values = [
                    paired_contrastive(
                        projected[left],
                        projected[right],
                        content_labels[normal],
                        args.contrastive_temperature,
                    )
                    for left, right in pairs
                    if left in content_n and right in content_n
                ]
                if values:
                    cross_modal = torch.stack(values).mean()
            retrieval_regularization = torch.zeros((), device=device)
            if model.retrieval_from_raw:
                retrieval_regularization = sum(
                    parameter.square().sum()
                    for head in model.retrieval_heads.values()
                    for parameter in head.parameters()
                )
            loss = (
                recon_loss
                + args.invariance_weight * invariance
                + args.alignment_weight * alignment
                + args.swap_weight * swap_loss
                + args.hsic_weight * hsic
                + args.content_weight * contrastive
                + cross_modal_weight * cross_modal
                + model.retrieval_regularization * retrieval_regularization
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            optimizer.step()
            for name in args.streams:
                unit_norm(model.content_encoders[name])
                unit_norm(model.nuisance_encoders[name])
                unit_norm(model.content_decoders[name])
                unit_norm(model.nuisance_decoders[name])
            values = {
                "loss": loss,
                "recon": recon_loss,
                "invariance": invariance,
                "alignment": alignment,
                "swap": swap_loss,
                "hsic": hsic,
                "contrastive": contrastive,
                "cross_modal": cross_modal,
            }
            for key, value in values.items():
                totals[key] += float(value.detach()) * len(normal)
        row = {
            "epoch": epoch,
            **{key: value / n for key, value in totals.items()},
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d} loss={row['loss']:.6f} "
                f"recon={row['recon']:.6f} inv={row['invariance']:.6f} "
                f"align={row['alignment']:.6f} swap={row['swap']:.6f} "
                f"hsic={row['hsic']:.5f} content={row['contrastive']:.4f} "
                f"cross={row['cross_modal']:.4f} "
                f"time={row['elapsed_s']:.1f}s"
            )
            if torch.cuda.is_available():
                row["peak_vram_gb"] = (
                    torch.cuda.max_memory_allocated() / 1e9
                )
    return model, history


def probe(rows, labels, mask):
    classifier = LogisticRegression(max_iter=700)
    classifier.fit(rows[mask], labels[mask])
    return float(classifier.score(rows[~mask], labels[~mask]))


def utility_scores(content_logits, train_contents, train_layouts):
    nuisance_var, content_var = variance_decomposition(
        content_logits,
        np.asarray(train_contents),
        np.asarray(train_layouts),
    )
    values = torch.log(
        (content_var + 1e-6) / (nuisance_var + 1e-6)
    )
    return (values - values.mean()) / (values.std() + 1e-6)


@torch.no_grad()
def evaluate(
    model,
    train_data,
    train_contents,
    train_layouts,
    test_data,
    test_contents,
    test_layouts,
    args,
    device,
):
    model.eval()
    train_logits, _ = model.encode_content(train_data)
    content_logits, nuisance_logits, content, nuisance = model(test_data)
    recon = model.decode(content, nuisance)
    contents = np.asarray(test_contents)
    layouts = np.asarray(test_layouts)
    labels = np.asarray([int(value) for value in layouts])
    mask = half_of(contents, seed=0)
    fitting, calibration, heldout = three_way_split(contents, seed=0)
    row = {
        "mean_content_R@1": metrics(
            torch.stack([
                content[name].float().cpu() for name in args.streams
            ]).mean(0),
            contents,
            layouts,
        )["NN_R@1"],
        "streams": {},
        "settings": {
            "variant": "joint" if args.joint_nuisance else "residual",
            "hsic_weight": args.hsic_weight,
            "nested_k": list(model.nested_k),
            "block_size": args.block_size,
            "content_k": args.content_k,
            "nuisance_k": args.nuisance_k,
            "epochs": args.epochs,
            "seed": args.seed,
        },
    }
    for name in args.streams:
        utility = utility_scores(
            train_logits[name].float().cpu(),
            train_contents,
            train_layouts,
        )
        logits = content_logits[name].float().cpu()
        rank = logits.argsort(dim=1).argsort(dim=1).float()
        rank = rank / max(1, logits.shape[1] - 1)
        candidates = []
        for strength in (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5):
            scores = rank + strength * utility
            _, indices = torch.topk(scores, args.content_k, dim=1)
            codes = torch.zeros_like(logits).scatter_(
                1, indices, F.relu(logits.gather(1, indices))
            )
            value = metrics(
                codes[calibration],
                contents[calibration],
                layouts[calibration],
            )["NN_R@1"]
            candidates.append((value, strength, codes))
        _, strength, farg_code = max(candidates, key=lambda item: item[0])
        dense_value = metrics(logits, contents, layouts)["NN_R@1"]
        raw_value = metrics(content[name].float().cpu(), contents, layouts)["NN_R@1"]
        farg_value = metrics(
            farg_code[heldout],
            contents[heldout],
            layouts[heldout],
        )["NN_R@1"]
        raw_heldout = metrics(
            content[name].float().cpu()[heldout],
            contents[heldout],
            layouts[heldout],
        )["NN_R@1"]
        row["streams"][name] = {
            "content_R@1": raw_value,
            "content_heldout_R@1": raw_heldout,
            "dense_R@1": dense_value,
            "farg_R@1": farg_value,
            "farg_strength": strength,
            "content_slots": float(
                (content[name] > 0).sum(1).float().mean()
            ),
            "nuisance_slots": float(
                (nuisance[name] > 0).sum(1).float().mean()
            ),
            "content_layout_probe": probe(
                content[name].float().cpu().numpy(), labels, mask
            ),
            "nuisance_layout_probe": probe(
                nuisance[name].float().cpu().numpy(), labels, mask
            ),
            "recon_mse": float(F.mse_loss(recon[name], test_data[name])),
            "dense_logit_std": float(content_logits[name].std()),
            "nuisance_logit_std": float(nuisance_logits[name].std()),
        }
    lookup, _, _ = build_lookup(contents, layouts)
    anchor, target, donor = random_triplets(
        lookup,
        min(512, len(contents)),
        torch.Generator().manual_seed(23),
    )
    anchor = anchor.to(device)
    target = target.to(device)
    donor = donor.to(device)
    _, _, content_a, _ = model(gather(test_data, anchor, args.streams))
    _, _, _, nuisance_d = model(gather(test_data, donor, args.streams))
    _, _, content_t, nuisance_t = model(gather(test_data, target, args.streams))
    swapped = model.decode(content_a, nuisance_d)
    normal = model.decode(content_t, nuisance_t)
    swaps, normals = {}, {}
    for name in args.streams:
        swaps[name] = float(
            F.mse_loss(swapped[name], test_data[name][target])
        )
        normals[name] = float(
            F.mse_loss(normal[name], test_data[name][target])
        )
    row["swap_mse"] = swaps
    row["normal_mse"] = normals
    row["swap_ratio"] = float(
        np.mean([
            swaps[name] / max(normals[name], 1e-12)
            for name in args.streams
        ])
    )
    return row


def load_caches(args):
    train = torch.load(args.train_cache, weights_only=False)
    test = torch.load(args.test_cache, weights_only=False)
    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Residual orthogonal factorized multi-stream SAE."
    )
    parser.add_argument("--streams", nargs="+", default=["clip", "siglip"])
    parser.add_argument(
        "--train-cache",
        default="outputs/cache_cifar100_full/cifar100_standard_train.pt",
    )
    parser.add_argument(
        "--test-cache",
        default="outputs/cache_cifar100_full/cifar100_standard_test.pt",
    )
    parser.add_argument("--content-latents", type=int, default=4096)
    parser.add_argument("--nuisance-latents", type=int, default=1024)
    parser.add_argument("--content-k", type=int, default=32)
    parser.add_argument("--nuisance-k", type=int, default=32)
    parser.add_argument("--nested-k", type=int, nargs="+", default=None)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--matryoshka-weight", type=float, default=0.25)
    parser.add_argument("--joint-nuisance", action="store_true")
    parser.add_argument("--hsic-weight", type=float, default=0.1)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/rof_sae.pt")
    parser.add_argument("--report-json", default="outputs/rof_sae.json")
    args = parser.parse_args()
    if args.nested_k is None:
        args.nested_k = [args.content_k]
    if args.joint_nuisance and args.block_size:
        raise ValueError("joint nuisance and block-sparse are separate controls")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_cache, test_cache = load_caches(args)
    contents = np.asarray(train_cache["contents"])
    layouts = np.asarray(train_cache["layouts"])
    train_data = {
        name: train_cache[name].to(device) for name in args.streams
    }
    test_data = {
        name: test_cache[name].to(device) for name in args.streams
    }
    model, history = train(
        args, train_data, contents, layouts, device
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        args.out,
    )
    report = evaluate(
        model,
        train_data,
        contents,
        layouts,
        test_data,
        test_cache["contents"],
        test_cache["layouts"],
        args,
        device,
    )
    report["history"] = history
    report["settings"].update({
        "train_rows": len(train_cache["contents"]),
        "test_rows": len(test_cache["contents"]),
        "classes": len(set(test_cache["labels"])),
        "streams": args.streams,
    })
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["streams"], indent=2))
    print(f"saved {args.out}")
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
