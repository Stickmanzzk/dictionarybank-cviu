"""Extract the three frozen feature streams used by DictionaryBank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DOMAIN_IDS = {
    "clipart": "0",
    "infograph": "1",
    "painting": "2",
    "quickdraw": "3",
    "real": "4",
    "sketch": "5",
}


class ImagePaths(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.records[index]["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt")[
                "pixel_values"
            ][0]
        return pixel_values


def discover(split_dir: Path, limit_per_class: int | None):
    records = []
    for domain_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        if domain_dir.name not in DOMAIN_IDS:
            continue
        for class_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if limit_per_class:
                paths = paths[:limit_per_class]
            records.extend(
                {
                    "path": str(path),
                    "domain": DOMAIN_IDS[domain_dir.name],
                    "class_name": class_dir.name,
                }
                for path in paths
            )
    if not records:
        raise RuntimeError(f"no DomainNet images found under {split_dir}")
    return records


def load_encoder(stream: str, model_id: str, device, local_files_only: bool):
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPVisionModel,
        SiglipVisionModel,
    )

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model_class = {
        "clip": CLIPVisionModel,
        "siglip": SiglipVisionModel,
        "dino": AutoModel,
    }[stream]
    model = model_class.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    ).to(device).eval()
    return model, processor


def extract(
    records,
    stream,
    model_id,
    processor,
    model,
    device,
    batch_size,
    workers,
):
    loader = DataLoader(
        ImagePaths(records, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    parts = []
    with torch.inference_mode():
        for pixel_values in tqdm(loader, desc=stream):
            outputs = model(pixel_values=pixel_values.to(device))
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = outputs.last_hidden_state[:, 0]
            parts.append(F.normalize(features.float(), dim=1).cpu())
    return torch.cat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records = discover(Path(args.split_dir), args.limit_per_class)
    class_names = sorted({record["class_name"] for record in records})
    if args.label_map:
        metadata = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
        class_names = metadata["classes"]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = torch.tensor(
        [class_to_index[record["class_name"]] for record in records],
        dtype=torch.long,
    )

    cache = {}
    for stream, model_id in (
        ("clip", args.clip_model),
        ("siglip", args.siglip_model),
        ("dino", args.dino_model),
    ):
        model, processor = load_encoder(
            stream,
            model_id,
            device,
            args.local_files_only,
        )
        cache[stream] = extract(
            records,
            stream,
            model_id,
            processor,
            model,
            device,
            args.batch_size,
            args.workers,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cache["labels"] = labels
    cache["domains"] = [record["domain"] for record in records]
    cache["contents"] = [record["path"] for record in records]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)

    metadata = {
        "dataset": "DomainNet",
        "split_dir": str(Path(args.split_dir)),
        "rows": len(records),
        "classes": class_names,
        "streams": {
            name: list(cache[name].shape)
            for name in ("clip", "siglip", "dino")
        },
        "model_ids": {
            "clip": args.clip_model,
            "siglip": args.siglip_model,
            "dino": args.dino_model,
        },
    }
    Path(args.metadata).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
