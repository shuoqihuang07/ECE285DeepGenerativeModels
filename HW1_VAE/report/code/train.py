"""Train a VAE on the Kaggle chest_xray directory (labels are ignored)."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from model import VAE, vae_loss
from plot_history import plot_history


class XRayDataset(Dataset):
    def __init__(self, root, image_size):
        root = Path(root)
        self.paths = sorted(p for p in root.rglob("*")
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                            and not p.name.startswith("._"))
        if not self.paths:
            raise ValueError(f"No images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("L"))


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, beta, optimizer=None):
    model.train(optimizer is not None)
    totals = dict.fromkeys(("loss", "reconstruction", "kl"), 0.0)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images in loader:
            images = images.to(device)
            logits, mu, logvar = model(images)
            losses = vae_loss(logits, images, mu, logvar, beta)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Non-finite loss; try a smaller learning rate")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
            count += images.size(0)
            for key, value in losses.items():
                totals[key] += value.item() * images.size(0)
    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train/ and val/ splits")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vae"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0 or args.beta < 0 or args.workers < 0:
        parser.error("epochs, batch-size, lr must be positive; beta and workers nonnegative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = {key: getattr(args, key) for key in ("image_size", "latent_dim", "base_channels")}
    model = VAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loaders = {
        split: DataLoader(XRayDataset(args.data_dir / split, args.image_size),
                          batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.workers, pin_memory=device.type == "cuda")
        for split in ("train", "val")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_images = next(iter(loaders["val"]))[:8].to(device)
    best_loss = float("inf")
    history = []
    print(f"Device: {device}; train: {len(loaders['train'].dataset)}; val: {len(loaders['val'].dataset)}")
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch}
        for split, loader in loaders.items():
            metrics = run_epoch(model, loader, device, args.beta,
                                optimizer if split == "train" else None)
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        history.append(row)
        print(f"Epoch {epoch:03d}: train={row['train_loss']:.3f}, val={row['val_loss']:.3f}")
        checkpoint = {"model_state_dict": model.state_dict(), "model_config": config,
                      "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                      "val_loss": row["val_loss"], "settings": settings}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        model.eval()
        with torch.no_grad():
            save_image(model.decode(fixed_z).sigmoid(),
                       args.output_dir / f"samples_{epoch:03d}.png", nrow=8)
            mu, _ = model.encode(fixed_images)
            reconstructed = model.decode(mu).sigmoid()
            save_image(torch.cat([fixed_images, reconstructed]),
                       args.output_dir / "reconstructions.png", nrow=len(fixed_images))
        with (args.output_dir / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
    plot_history(history, args.output_dir / "loss_curves.png")


if __name__ == "__main__":
    main()
