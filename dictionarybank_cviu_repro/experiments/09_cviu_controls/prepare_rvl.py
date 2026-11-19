import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train = torch.load(args.train, map_location="cpu", weights_only=False)
    test = torch.load(args.test, map_location="cpu", weights_only=False)
    streams = [name for name in ("clip", "siglip", "dino") if name in train]
    combined = {
        name: torch.cat([train[name], test[name]], dim=0)
        for name in streams
    }
    combined["labels"] = torch.cat(
        [train["labels"], test["labels"]], dim=0
    )
    combined["domains"] = (
        ["source"] * len(train["labels"])
        + ["target"] * len(test["labels"])
    )
    if "contents" in train:
        combined["contents"] = list(train["contents"]) + list(test["contents"])
    if "layouts" in train:
        combined["layouts"] = torch.cat(
            [
                torch.as_tensor(train["layouts"]),
                torch.as_tensor(test["layouts"]),
            ],
            dim=0,
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, output)
    print({
        "path": str(output),
        "train": len(train["labels"]),
        "test": len(test["labels"]),
        "classes": len(set(np.asarray(train["labels"]).tolist())),
        "streams": streams,
    })


if __name__ == "__main__":
    main()
