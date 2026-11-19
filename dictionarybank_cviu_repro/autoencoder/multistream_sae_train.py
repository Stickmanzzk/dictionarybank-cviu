import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import torch

from experiment_utils import metrics


class MultiStreamSAE(torch.nn.Module):
    def __init__(self, widths, n_latents, k):
        super().__init__()
        self.names = list(widths)
        self.n_latents = n_latents
        self.k = k
        self.encoders = torch.nn.ModuleDict({
            name: torch.nn.Linear(widths[name], n_latents, bias=True)
            for name in self.names})
        self.pre_bias = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(widths[name]))
            for name in self.names})

    def forward(self, streams):
        logits = {name: self.encoders[name](streams[name] - self.pre_bias[name])
                  for name in self.names}
        stacked = torch.cat([logits[name] for name in self.names], dim=1)
        values, indices = torch.topk(stacked, self.k, dim=1)
        sparse = {name: torch.zeros_like(logits[name]) for name in self.names}
        offset = 0
        for name in self.names:
            mask = (indices >= offset) & (indices < offset + self.n_latents)
            local = (indices - offset).clamp(0, self.n_latents - 1)
            selected = torch.where(mask, torch.relu(values), torch.zeros_like(values))
            sparse[name].scatter_(1, local, selected)
            offset += self.n_latents
        recon = {name: sparse[name] @ self.encoders[name].weight + self.pre_bias[name]
                 for name in self.names}
        return logits, sparse, recon


def unit_norm(module):
    with torch.no_grad():
        module.weight.div_(module.weight.norm(dim=1, keepdim=True) + 1e-8)


def topk(logits, k):
    values, indices = torch.topk(logits, k, dim=1)
    out = torch.zeros_like(logits)
    out.scatter_(1, indices, torch.relu(values))
    return out


def main():
    parser = argparse.ArgumentParser(description="Train a second multi-stream SAE "
                                                 "on the disjoint training grid.")
    parser.add_argument("--train-clip", default="outputs/cache/sae_train_clip_openai.pt")
    parser.add_argument("--train-dino", default="outputs/cache/sae_train_dinov2_l.pt")
    parser.add_argument("--eval-clip", default="outputs/cache/documents_clip_openai.pt")
    parser.add_argument("--eval-dino", default="outputs/cache/documents_dinov2_l.pt")
    parser.add_argument("--n-latents", type=int, default=4096)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text", choices=("zero", "real"), default="zero")
    parser.add_argument("--clip-weights", default="clip_weights/open_clip_pytorch_model.bin")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/multistream_sae.pt")
    parser.add_argument("--report-json", default="outputs/multistream_sae.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_clip = torch.load(args.train_clip, weights_only=False)
    train_dino = torch.load(args.train_dino, weights_only=False)
    clip = train_clip["features"]
    dino = train_dino["features"]
    n = clip.shape[0]
    if args.text == "real":
        import open_clip
        from topic_layout_probe import load_corpus
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=args.clip_weights, device=device)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        corpus = load_corpus("dbpedia")
        texts = []
        for content in train_clip["contents"]:
            topic, index = content.split("::")
            texts.append(corpus[topic][int(index)])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                tokens = tokenizer(texts[start:start + 64]).to(device)
                vectors = clip_model.encode_text(tokens).float()
                chunks.append((vectors / vectors.norm(dim=-1, keepdim=True)).cpu())
        text_tensor = torch.cat(chunks, 0)
        del clip_model
        torch.cuda.empty_cache()
    else:
        text_tensor = torch.zeros(n, 768)
    print(f"train rows={n} clip={tuple(clip.shape)} dino={tuple(dino.shape)}")

    model = MultiStreamSAE({"clip_img": 768, "dino": 1024, "clip_txt": 768},
                           args.n_latents, args.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    streams = {
        "clip_img": clip.to(device),
        "dino": dino.to(device),
        "clip_txt": text_tensor.to(device),
    }
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = {name: value[order[start:start + args.batch_size]]
                     for name, value in streams.items()}
            logits, sparse, recon = model(batch)
            loss = sum(torch.nn.functional.mse_loss(recon[name], batch[name])
                       for name in batch)
            total += float(loss) * batch["clip_img"].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name in model.names:
                unit_norm(model.encoders[name])
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  mse={total / n:.5f}")

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")

    eval_clip = torch.load(args.eval_clip, weights_only=False)
    eval_dino = torch.load(args.eval_dino, weights_only=False)
    contents, layouts = eval_clip["contents"], eval_clip["layouts"]
    with torch.no_grad():
        batch = {
            "clip_img": eval_clip["features"].to(device),
            "dino": eval_dino["features"].to(device),
            "clip_txt": torch.zeros(len(contents), 768, device=device),
        }
        logits, sparse, _ = model(batch)

    report = {}
    for name in model.names:
        slots = (sparse[name] > 0).sum(dim=1)
        report[name] = {
            "mean_slots": float(slots.float().mean()),
            "dense_R@1": metrics(logits[name].cpu(), contents, layouts)["NN_R@1"],
            "released_R@1": metrics(sparse[name].cpu(), contents, layouts)["NN_R@1"],
        }
        print(f"{name:9s} slots {report[name]['mean_slots']:5.1f}  "
              f"dense {report[name]['dense_R@1']:.3f}  "
              f"released {report[name]['released_R@1']:.3f}")
    report["clip_budget"] = {
        f"q={q}": metrics(topk(logits["clip_img"], q), contents, layouts)["NN_R@1"]
        for q in (8, 16, 24, 32, 48, 64)}
    print("clip budget: " + "  ".join(f"{k}:{v:.3f}"
                                      for k, v in report["clip_budget"].items()))
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
