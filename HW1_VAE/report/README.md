# HW1 VAE report package

`report.pdf` is the two-page report, including references, built with the official
NeurIPS 2026 style in preprint mode. `source/report.tex` is editable LaTeX.
The author field currently identifies the course/homework; add your name if needed.

## Contents

- `source/`: report text, official unmodified `neurips_2026.sty`, and measured
  values in `results.tex`.
- `figures/`: training curves, generated samples, real test examples, and
  posterior-mean reconstructions. The report uses `training_curves.png`,
  `generated_grid.png`, and `real_test_grid.png`.
- `data/history.csv`, `config.json`, `run_summary.json`: all 50 training epochs,
  original settings, split counts, selected checkpoint epoch, and checkpoint hashes.
- `data/generated/`: the exact 624 generated PNGs used for evaluation.
- `data/real_test/`: the exact 624 held-out images after grayscale conversion and
  resizing to 128x128, saved as PNGs. These are evaluation inputs, not the original
  full-resolution Kaggle files.
- `data/evaluation_manifest.csv`: source test filenames and SHA-256 hashes of
  original files and both sets of evaluation PNGs.
- `data/dataset_manifest.csv`: all 5,856 source filenames and split/class labels.
- `data/inception_outputs.npz`: real/generated 2,048-D features and 1,000-way
  probabilities, enabling offline score recomputation.
- `data/feature_metadata.json`, `metrics.json`: full metric protocol, exact software
  versions, pretrained weight hash, raw scores, and per-split IS values.
- `checkpoints/best.pt`: the selected epoch-45 checkpoint with optimizer tensors
  removed to reduce size. Model weights, configuration, epoch, validation loss,
  and original settings are preserved. The original training checkpoint is unchanged.
- `code/`: a snapshot of the model, training, generation, plotting, and tests,
  plus report asset preparation and Ignite evaluation code.
- `checksums.sha256`: hashes of all packaged deliverables except this checksum file.

Training images and Inception weights are not duplicated here. All inputs needed
to rebuild the report and recompute its scores are included; repeating training
requires the original Kaggle dataset.

## Setup

From this `report/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` records the exact evaluation environment. The portable
requirements above allow a compatible PyTorch/torchvision pair for your platform.
PDF compilation additionally requires Tectonic or a LaTeX installation.

## Recompute scores offline

```bash
python code/evaluate.py --recompute
```

This runs Ignite's `FID` and `InceptionScore` on the saved feature arrays. No
dataset download, model inference, or network access is needed. Matrix square
roots can take a minute. `metrics.json` is regenerated from the saved outputs.

## Re-extract features from the exact saved images

```bash
python code/evaluate.py --device cpu --batch-size 16
```

On first use, torchvision downloads the ImageNet Inception-v3 weights. Both real
and generated inputs are the bundled 128x128 8-bit PNGs. They are converted to
RGB (three identical channels), resized directly to 299x299 with PIL bilinear
interpolation, and normalized by ImageNet mean/std. The pretrained torchvision
model also applies its built-in `transform_input=True` conversion. One forward
pass captures the average-pool features and classification probabilities.

The evaluator adapts the [assigned Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)
using an Ignite `Engine`, `FID`, `InceptionScore`, and PIL interpolation. Changes
are explicit: held-out chest X-rays, consistent ImageNet normalization, no dropped
final batch, cached features, and **2,048-D pooled features for the main FID**.
This is torchvision-based FID, not TensorFlow-compatible `pytorch-fid`/Clean-FID;
scores should only be compared under the same protocol and sample count.

The installed Ignite default uses 1,000-way classifier probabilities for FID.
An additional `fid_probability1000` value is saved as a diagnostic with the same
normalized inputs; it is not the pooled-feature FID in the report and is not an
exact rerun of the older tutorial's unnormalized GAN preprocessing.

The main IS uses all 624 generated examples as one set. The ten-split mean/SD is
also recorded (four splits of 63 images and six of 62); this SD is not a confidence
interval. Real-test IS is context only. Inception is trained on ImageNet rather
than medical labels; these scores do not establish clinical image quality.

## Rebuild figures and PDF

```bash
python code/build_report_assets.py
cd source
tectonic --outdir .. report.tex
```

Alternatively, run `pdflatex -output-directory=.. report.tex` twice from `source/`
or upload `source/` and `figures/` together to Overleaf, preserving relative paths.
The official template is available from the [NeurIPS 2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers).
Report margins and body font sizes use the official style settings.

## Generate additional images from the packaged checkpoint

```bash
python code/generate.py --checkpoint checkpoints/best.pt --num-samples 64 --seed 2026 --output-dir additional_samples --device cpu
```

For the exact 624-image sequence used in the report, use `--num-samples 624
--batch-size 32 --seed 2026 --device cpu`. Generation batch size can affect the
sequence of random draws, so preserve it for exact reproduction.

`python code/prepare_assets.py` recreates the original bundle inputs from the
parent homework directory's `runs/vae/best.pt` and `data/chest_xray`. It is only
needed to rebuild the input bundle, and overwrites the corresponding saved assets.

To repeat training, download the [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and run `code/train.py` with `--data-dir` pointing to its train/val/test parent.
The exact training hyperparameters are in `data/config.json`. The original run
used automatic device selection; the checkpoint does not record the resolved
training device or original library versions. Only the evaluation environment
versions are known and recorded.

## Validation

```bash
python -m unittest discover -s code -p 'test_*.py' -v
python code/verify_package.py
```

The verification checks all packaged hashes, exact PNG counts and image hashes,
feature/probability dimensions, finite scores, and the two-page PDF. The PDF was
also rendered and visually inspected before delivery. Regenerate the checksum
manifest with `python code/verify_package.py --write-checksums` after intentional
edits or rebuilding outputs.
