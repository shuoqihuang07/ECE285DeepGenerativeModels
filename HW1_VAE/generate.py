"""Sample grayscale X-rays from a trained VAE checkpoint."""

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import VAE
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size) < 1:
        parser.error("num-samples and batch-size must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview = []
    for start in range(0, args.num_samples, args.batch_size):
        images = model.sample(min(args.batch_size, args.num_samples - start)).cpu()
        for index, image in enumerate(images, start=start):
            save_image(image, args.output_dir / f"sample_{index:05d}.png")
            if len(preview) < 64:
                preview.append(image)
    save_image(torch.stack(preview), args.output_dir / "grid.png",
               nrow=min(8, math.ceil(math.sqrt(len(preview)))))
    print(f"Saved {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
