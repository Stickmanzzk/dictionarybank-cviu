import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class BudgetBank(torch.nn.Module):
    def __init__(self, width, members, latent, active_budget):
        super().__init__()
        self.latent = members * latent
        self.member_latent = latent
        self.group_k = active_budget
        self.weight = torch.nn.Parameter(
            torch.empty(self.latent, width),
            requires_grad=False,
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self.latent),
            requires_grad=False,
        )

    def forward(self, values):
        logits = F.linear(values, self.weight, self.bias)
        selected, indices = torch.topk(logits, self.group_k, dim=1)
        code = torch.zeros_like(logits).scatter_(
            1,
            indices,
            torch.relu(selected),
        )
        return code


def build_models(paths, streams, active_budget, device):
    blobs = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    saved = blobs[0]["args"]
    members = len(blobs)
    latent = int(saved["content_latents"])
    models = {}
    for name in streams:
        weights = []
        biases = []
        for blob in blobs:
            state = blob["state_dict"]
            weight = state[f"content_encoders.{name}.weight"].float()
            bias = state[f"content_encoders.{name}.bias"].float()
            pre_bias = state[f"pre_bias.{name}"].float()
            weights.append(weight)
            biases.append(bias - weight @ pre_bias)
        model = BudgetBank(
            weights[0].shape[1],
            members,
            latent,
            active_budget,
        )
        with torch.no_grad():
            model.weight.copy_(torch.cat(weights, 0))
            model.bias.copy_(torch.cat(biases, 0))
        models[name] = model.to(device).eval()
    return models


def entropy(counts, members):
    probability = counts / max(int(counts.sum()), 1)
    positive = probability[probability > 0]
    if len(positive) <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / np.log(members))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--checkpoints", nargs=3, required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--active-budget", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(
        args.target_cache,
        map_location="cpu",
        weights_only=False,
    )
    payload = torch.load(
        args.prediction,
        map_location="cpu",
        weights_only=False,
    )
    rows = np.flatnonzero(
        np.asarray(cache["domains"]) == args.target_domain
    )
    labels = torch.tensor(
        np.asarray(cache["labels"])[rows],
        dtype=torch.long,
    )
    final_correct = (
        payload["target_probability"].argmax(1).eq(labels).numpy()
    )
    models = build_models(
        args.checkpoints,
        ("clip", "siglip", "dino"),
        args.active_budget,
        device,
    )

    output = {
        "rows": int(len(rows)),
        "active_budget": args.active_budget,
        "streams": {},
    }
    for stream, model in models.items():
        member_counts = np.zeros(3, dtype=np.int64)
        entropies = []
        max_shares = []
        energy_entropies = []
        selected_correct = []
        selected_incorrect = []
        used = np.zeros(model.latent, dtype=bool)
        for start in range(0, len(rows), args.eval_batch_size):
            batch_rows = rows[start:start + args.eval_batch_size]
            batch = cache[stream][batch_rows].to(device)
            logits = F.linear(batch, model.weight, model.bias)
            values, indices = torch.topk(
                logits,
                args.active_budget,
                dim=1,
            )
            values = torch.relu(values)
            cpu_indices = indices.cpu().numpy()
            cpu_values = values.cpu().numpy()
            used[cpu_indices.ravel()] = True
            local_correct = final_correct[
                start:start + len(batch_rows)
            ]
            for row_indices, row_values, is_correct in zip(
                cpu_indices,
                cpu_values,
                local_correct,
            ):
                members = row_indices // model.member_latent
                counts = np.bincount(members, minlength=3)
                member_counts += counts
                sample_entropy = entropy(counts, 3)
                entropies.append(sample_entropy)
                max_shares.append(float(counts.max() / counts.sum()))
                block_energy = np.array([
                    np.square(row_values[members == member]).sum()
                    for member in range(3)
                ])
                energy_entropy = entropy(block_energy, 3)
                energy_entropies.append(energy_entropy)
                target = (
                    selected_correct if is_correct else selected_incorrect
                )
                target.append(sample_entropy)

        entropies_np = np.asarray(entropies)
        quantiles = np.quantile(entropies_np, [0.0, 0.25, 0.5, 0.75, 1.0])
        bins = []
        for index in range(4):
            low = quantiles[index]
            high = quantiles[index + 1]
            mask = (
                (entropies_np >= low)
                & (
                    entropies_np <= high
                    if index == 3
                    else entropies_np < high
                )
            )
            bins.append({
                "low": float(low),
                "high": float(high),
                "rows": int(mask.sum()),
                "accuracy": float(final_correct[mask].mean())
                if mask.any()
                else None,
            })
        output["streams"][stream] = {
            "member_selection_fraction": (
                member_counts / member_counts.sum()
            ).tolist(),
            "mean_support_entropy": float(np.mean(entropies_np)),
            "median_support_entropy": float(np.median(entropies_np)),
            "mean_max_member_share": float(np.mean(max_shares)),
            "member_switch_fraction": float(
                np.mean(np.asarray(max_shares) < 0.5)
            ),
            "mean_block_energy_entropy": float(np.mean(energy_entropies)),
            "support_entropy_correct": float(np.mean(selected_correct)),
            "support_entropy_incorrect": float(np.mean(selected_incorrect))
            if selected_incorrect
            else None,
            "unique_atoms_used": int(used.sum()),
            "total_atoms": int(model.latent),
            "unique_atom_fraction": float(used.mean()),
            "entropy_quantiles": bins,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
