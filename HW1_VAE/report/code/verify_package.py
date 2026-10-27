"""Verify report inputs, results, PDF page count, and package checksums."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader

REPORT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def package_files():
    return sorted(p for p in REPORT.rglob('*') if p.is_file()
                  and p.name != 'checksums.sha256'
                  and not any(part.startswith('.') or part == '__pycache__'
                              for part in p.relative_to(REPORT).parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-checksums', action='store_true')
    args = parser.parse_args()
    data = REPORT / 'data'
    summary = json.loads((data / 'run_summary.json').read_text())
    results = json.loads((data / 'metrics.json').read_text())
    n = summary['num_generated']
    assert n == results['protocol']['num_generated'] == results['protocol']['num_real']
    assert sha256(REPORT / 'checkpoints/best.pt') == summary['inference_checkpoint_sha256']
    manifest = list(csv.DictReader((data / 'evaluation_manifest.csv').open()))
    assert len(manifest) == n
    for group, prefix, key in (('real_test', 'real', 'real_png_sha256'),
                               ('generated', 'sample', 'generated_png_sha256')):
        files = [p for p in (data / group).glob('*.png') if not p.name.startswith('._')]
        assert len(files) == n
        for row in manifest:
            path = data / group / f'{prefix}_{int(row["index"]):05d}.png'
            assert sha256(path) == row[key], f'Image hash mismatch: {path}'
            with Image.open(path) as image:
                assert image.size == (128, 128)
                image.verify()
    with np.load(data / 'inception_outputs.npz') as arrays:
        for key in ('real_features', 'generated_features'):
            assert arrays[key].shape == (n, 2048)
            assert np.isfinite(arrays[key]).all()
        for key in ('real_probs', 'generated_probs'):
            assert arrays[key].shape == (n, 1000)
            assert (arrays[key] >= 0).all()
            assert np.allclose(arrays[key].sum(1), 1)
    metrics = results['metrics']
    assert all(np.isfinite(value).all() for value in metrics.values())
    assert metrics['fid_pool2048'] >= 0 and metrics['inception_score'] >= 1
    pdf = PdfReader(REPORT / 'report.pdf')
    assert len(pdf.pages) == 2, f'Expected 2 pages, found {len(pdf.pages)}'
    text = '\n'.join(page.extract_text() for page in pdf.pages)
    for value in (f'{metrics["fid_pool2048"]:.2f}', f'{metrics["inception_score"]:.3f}'):
        assert value in text, f'Measured result missing from PDF: {value}'
    checksum_path = REPORT / 'checksums.sha256'
    if args.write_checksums:
        checksum_path.write_text(''.join(f'{sha256(p)}  {p.relative_to(REPORT).as_posix()}\n'
                                          for p in package_files()))
    entries = [line.split('  ', 1) for line in checksum_path.read_text().splitlines()]
    assert {name for _, name in entries} == {p.relative_to(REPORT).as_posix() for p in package_files()}
    for expected, name in entries:
        assert sha256(REPORT / name) == expected, f'Package checksum mismatch: {name}'
    print(f'PASS: {n} real + {n} generated images, feature arrays, scores, 2-page PDF, {len(entries)} file checksums')


if __name__ == '__main__':
    main()
