"""Ignite FID/IS evaluation adapted from the assignment's Ignite tutorial.

https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
Uses a held-out test set, PIL resizing and an Ignite evaluator. FID explicitly
uses pooled features rather than the tutorial's default classifier outputs.
All exact PNG inputs and extracted features are retained for audit/recomputation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import functional as TF

REPORT = Path(__file__).resolve().parents[1]


class EvaluationImages(Dataset):
    def __init__(self, root):
        self.paths = sorted(p for p in root.glob('*.png') if not p.name.startswith('._'))
        if not self.paths:
            raise ValueError(f'No evaluation PNGs in {root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            # Both sets start from the saved 128x128, 8-bit grayscale PNGs.
            image = image.convert('RGB').resize((299, 299), Image.Resampling.BILINEAR)
            return TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def extract(model, dataset, batch_size, device):
    features, probabilities = [], []
    capture = {}
    handle = model.avgpool.register_forward_hook(lambda _m, _i, out: capture.update(pool=out))
    try:
        with torch.inference_mode():
            for index, images in enumerate(DataLoader(dataset, batch_size=batch_size)):
                logits = model(images.to(device))
                features.append(capture['pool'].flatten(1).cpu().numpy())
                probabilities.append(logits.softmax(1).cpu().numpy())
                print(f'  {min((index + 1) * batch_size, len(dataset))}/{len(dataset)}', flush=True)
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(probabilities)


def compute_metrics(real_features, generated_features, real_probs, generated_probs, batch_size):
    if len(real_features) != len(generated_features) or len(real_features) < 2:
        raise ValueError('FID requires equal real/generated set sizes, each at least two')
    # Identity extractors let Ignite consume our saved Inception outputs exactly.
    evaluator = Engine(lambda _engine, batch: batch)
    FID(num_features=2048, output_transform=lambda out: (out[0], out[1])).attach(evaluator, 'fid_pool2048')
    InceptionScore(num_features=1000, output_transform=lambda out: out[2]).attach(evaluator, 'inception_score')
    InceptionScore(num_features=1000, output_transform=lambda out: out[3]).attach(evaluator, 'real_inception_score')
    # Extra diagnostic matching the classifier-probability feature choice of the
    # installed Ignite default (same normalized inputs as the pooled-feature run).
    FID(num_features=1000, output_transform=lambda out: (out[2], out[3])).attach(evaluator, 'fid_probability1000')
    batches = [tuple(torch.from_numpy(array[start:start + batch_size]) for array in
                     (generated_features, real_features, generated_probs, real_probs))
               for start in range(0, len(real_features), batch_size)]
    print('Computing Ignite metrics (matrix square roots may take a minute)...', flush=True)
    metrics = {key: float(value) for key, value in evaluator.run(batches).metrics.items()}
    split_scores = []
    for probabilities in np.array_split(generated_probs, 10):
        metric = InceptionScore(num_features=1000)
        metric.update(torch.from_numpy(probabilities))
        split_scores.append(metric.compute())
    metrics.update(is_10split_mean=float(np.mean(split_scores)),
                   is_10split_std=float(np.std(split_scores)), is_10split_values=split_scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--recompute', action='store_true', help='Reuse saved features, without downloading weights')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    data = REPORT / 'data'
    feature_file = data / 'inception_outputs.npz'
    metadata_file = data / 'feature_metadata.json'
    if args.recompute:
        with np.load(feature_file) as arrays:
            real_features, generated_features, real_probs, generated_probs = (
                arrays[key] for key in ('real_features', 'generated_features', 'real_probs', 'generated_probs'))
        metadata = json.loads(metadata_file.read_text())
    else:
        device = torch.device(args.device)
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights).to(device).eval()
        print('Extracting real test features...', flush=True)
        real = EvaluationImages(data / 'real_test')
        fake = EvaluationImages(data / 'generated')
        real_features, real_probs = extract(model, real, args.batch_size, device)
        print('Extracting generated features...', flush=True)
        generated_features, generated_probs = extract(model, fake, args.batch_size, device)
        np.savez_compressed(feature_file, real_features=real_features, generated_features=generated_features,
                            real_probs=real_probs, generated_probs=generated_probs)
        weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
        metadata = {'weights': str(weights), 'weights_url': weights.url,
                    'weights_sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                    'num_real': len(real), 'num_generated': len(fake), 'device': str(device),
                    'batch_size': args.batch_size, 'threads': args.threads,
                    'feature_dimension': 2048, 'probability_dimension': 1000,
                    'preprocessing': '128x128 saved PNG -> RGB -> PIL bilinear 299x299 -> ToTensor [0,1] -> ImageNet mean/std; torchvision transform_input=True',
                    'fid_implementation': 'ignite.metrics.FID, num_features=2048, identity extractor on avgpool outputs',
                    'additional_fid': '1000-way softmax probability FID, not standard pool-feature FID',
                    'source': 'https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/',
                    'platform': platform.platform(), 'versions': {package: importlib.metadata.version(package)
                                for package in ('torch', 'torchvision', 'pytorch-ignite', 'numpy', 'scipy', 'Pillow')}}
        metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')
    metrics = compute_metrics(real_features, generated_features, real_probs, generated_probs, args.batch_size)
    results = {'metrics': metrics, 'protocol': metadata}
    (data / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
