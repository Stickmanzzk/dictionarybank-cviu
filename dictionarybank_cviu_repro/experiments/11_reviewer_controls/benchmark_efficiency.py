import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def load_stream_weights(paths, stream):
    weights = []
    biases = []
    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["state_dict"]
        weight = state[f"content_encoders.{stream}.weight"].float()
        bias = state[f"content_encoders.{stream}.bias"].float()
        pre_bias = state[f"pre_bias.{stream}"].float()
        weights.append(weight)
        biases.append(bias - weight @ pre_bias)
    return torch.cat(weights, 0), torch.cat(biases, 0)


def measure_bank(weights, features, prototypes, active_budget, repeats):
    for _ in range(10):
        for stream, (weight, bias) in weights.items():
            logits = F.linear(features[stream], weight, bias)
            values, indices = torch.topk(logits, active_budget, dim=1)
            code = torch.zeros_like(logits).scatter_(
                1,
                indices,
                torch.relu(values),
            )
            F.normalize(code, dim=1) @ prototypes[stream].T
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        for stream, (weight, bias) in weights.items():
            logits = F.linear(features[stream], weight, bias)
            values, indices = torch.topk(logits, active_budget, dim=1)
            code = torch.zeros_like(logits).scatter_(
                1,
                indices,
                torch.relu(values),
            )
            F.normalize(code, dim=1) @ prototypes[stream].T
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def measure_separate(groups, features, prototypes, k, repeats):
    for _ in range(5):
        for stream in features:
            latent = groups[stream][0][0].shape[0]
            for group, (weight, bias) in enumerate(groups[stream]):
                logits = F.linear(features[stream], weight, bias)
                values, indices = torch.topk(logits, k, dim=1)
                code = torch.zeros_like(logits).scatter_(
                    1,
                    indices,
                    torch.relu(values),
                )
                block = prototypes[stream][
                    :, group * latent:(group + 1) * latent
                ]
                F.normalize(code, dim=1) @ block.T
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        for stream in features:
            latent = groups[stream][0][0].shape[0]
            for group, (weight, bias) in enumerate(groups[stream]):
                logits = F.linear(features[stream], weight, bias)
                values, indices = torch.topk(logits, k, dim=1)
                code = torch.zeros_like(logits).scatter_(
                    1,
                    indices,
                    torch.relu(values),
                )
                block = prototypes[stream][
                    :, group * latent:(group + 1) * latent
                ]
                F.normalize(code, dim=1) @ block.T
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", nargs=3, required=True)
    parser.add_argument("--wide", required=True)
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["clip", "siglip", "dino"],
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--classes", type=int, default=344)
    parser.add_argument("--active-budget", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    bank = {
        stream: tuple(
            value.to(device)
            for value in load_stream_weights(args.bank, stream)
        )
        for stream in args.streams
    }
    wide = {
        stream: tuple(
            value.to(device)
            for value in load_stream_weights([args.wide], stream)
        )
        for stream in args.streams
    }
    separate = {
        stream: [
            tuple(
                value.to(device)
                for value in load_stream_weights([path], stream)
            )
            for path in args.bank
        ]
        for stream in args.streams
    }
    features = {
        stream: torch.randn(
            args.batch_size,
            bank[stream][0].shape[1],
            device=device,
        )
        for stream in args.streams
    }
    prototypes = {
        stream: torch.randn(
            args.classes,
            bank[stream][0].shape[0],
            device=device,
        )
        for stream in args.streams
    }

    measurements = {
        "bank": [],
        "wide": [],
        "separate_checkpoints": [],
    }
    for _ in range(args.trials):
        measurements["bank"].append(
            measure_bank(
                bank,
                features,
                prototypes,
                args.active_budget,
                args.repeats,
            )
        )
        measurements["wide"].append(
            measure_bank(
                wide,
                features,
                prototypes,
                args.active_budget,
                args.repeats,
            )
        )
        measurements["separate_checkpoints"].append(
            measure_separate(
                separate,
                features,
                prototypes,
                args.active_budget,
                args.repeats,
            )
        )

    summary = {}
    for name, values in measurements.items():
        values = torch.tensor(values)
        summary[name] = {
            "mean_ms": float(values.mean()),
            "std_ms": float(values.std(unbiased=True)),
            "trials": values.tolist(),
        }

    torch.cuda.reset_peak_memory_stats()
    measure_bank(
        bank,
        features,
        prototypes,
        args.active_budget,
        1,
    )
    peak_bank = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    measure_bank(
        wide,
        features,
        prototypes,
        args.active_budget,
        1,
    )
    peak_wide = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    measure_separate(
        separate,
        features,
        prototypes,
        args.active_budget,
        1,
    )
    peak_separate = int(torch.cuda.max_memory_allocated())

    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "active_budget": args.active_budget,
        "latency": summary,
        "bank_speedup_vs_separate": (
            summary["separate_checkpoints"]["mean_ms"]
            / summary["bank"]["mean_ms"]
        ),
        "bank_vs_wide_ratio": (
            summary["bank"]["mean_ms"]
            / summary["wide"]["mean_ms"]
        ),
        "peak_allocated_bytes": {
            "bank": peak_bank,
            "wide": peak_wide,
            "separate_checkpoints": peak_separate,
        },
        "parameters": {
            "bank": int(
                sum(
                    weight.numel() + bias.numel()
                    for weight, bias in bank.values()
                )
            ),
            "wide": int(
                sum(
                    weight.numel() + bias.numel()
                    for weight, bias in wide.values()
                )
            ),
        },
        "checkpoint_bytes": {
            "bank_members": [os.path.getsize(path) for path in args.bank],
            "bank_total": sum(os.path.getsize(path) for path in args.bank),
            "wide": os.path.getsize(args.wide),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
