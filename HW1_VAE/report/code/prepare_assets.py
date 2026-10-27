"""Freeze the trained run and build the exact image sets used in the report."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import VAE
from plot_history import plot_history
from train import XRayDataset

REPORT = Path(__file__).resolve().parents[1]
HW = REPORT.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(6)
    torch.manual_seed(2026)
    run = HW / 'runs/vae'
    data = REPORT / 'data'
    figures = REPORT / 'figures'
    for directory in (data, figures, REPORT / 'checkpoints'):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'history.csv'):
        shutil.copy2(run / name, data / name)
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model = VAE(**checkpoint['model_config']).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    # Retain all inference information, without duplicating Adam's optimizer tensors.
    torch.save({k: v for k, v in checkpoint.items() if k != 'optimizer_state_dict'},
               REPORT / 'checkpoints/best.pt')
    history = list(csv.DictReader((data / 'history.csv').open()))
    plot_history(history, figures / 'loss_curves.png')
    counts = {}
    manifest = []
    for split in ('train', 'val', 'test'):
        ds = XRayDataset(HW / 'data/chest_xray' / split, model.image_size)
        counts[split] = {'total': len(ds), **{
            c: sum(p.parent.name == c for p in ds.paths) for c in ('NORMAL', 'PNEUMONIA')}}
        manifest.extend({'split': split, 'class': p.parent.name,
                         'source_path': str(p.relative_to(HW / 'data/chest_xray'))}
                        for p in ds.paths)
    with (data / 'dataset_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'class', 'source_path'])
        writer.writeheader()
        writer.writerows(manifest)
    test = XRayDataset(HW / 'data/chest_xray/test', model.image_size)
    for group in ('real_test', 'generated'):
        (data / group).mkdir(exist_ok=True)
    image_manifest = []
    preview_real, preview_fake = [], []
    for start in range(0, len(test), 32):
        size = min(32, len(test) - start)
        fake = model.sample(size)
        for j in range(size):
            index = start + j
            real = test[index]
            real_path = data / 'real_test' / f'real_{index:05d}.png'
            fake_path = data / 'generated' / f'sample_{index:05d}.png'
            save_image(real, real_path)
            save_image(fake[j], fake_path)
            image_manifest.append({
                'index': index, 'source_path': str(test.paths[index].relative_to(HW / 'data/chest_xray')),
                'source_sha256': sha256(test.paths[index]),
                'real_png_sha256': sha256(real_path), 'generated_png_sha256': sha256(fake_path)})
            if index < 16:
                preview_real.append(real)
                preview_fake.append(fake[j])
    save_image(torch.stack(preview_real), figures / 'real_test_grid.png', nrow=4)
    save_image(torch.stack(preview_fake), figures / 'generated_grid.png', nrow=4)
    with torch.no_grad():
        originals = torch.stack(preview_real[:8])
        mu, _ = model.encode(originals)
        save_image(torch.cat((originals, model.decode(mu).sigmoid())),
                   figures / 'reconstruction_grid.png', nrow=8)
    with (data / 'evaluation_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_manifest[0]))
        writer.writeheader()
        writer.writerows(image_manifest)
    best_row = history[checkpoint['epoch'] - 1]
    summary = {'counts': counts, 'num_parameters': sum(p.numel() for p in model.parameters()),
               'selected_epoch': checkpoint['epoch'], 'selected_val_loss': checkpoint['val_loss'],
               'first_epoch': history[0], 'last_epoch': history[-1], 'selected_epoch_history': best_row,
               'generation_seed': 2026, 'generation_batch_size': 32,
               'num_generated': len(test), 'original_checkpoint_sha256': sha256(run / 'best.pt'),
               'inference_checkpoint_sha256': sha256(REPORT / 'checkpoints/best.pt'),
               'generation_device': 'cpu', 'model_config': checkpoint['model_config'],
               'training_settings': checkpoint['settings']}
    (data / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('counts', 'selected_epoch', 'num_generated')}, indent=2))


if __name__ == '__main__':
    main()
