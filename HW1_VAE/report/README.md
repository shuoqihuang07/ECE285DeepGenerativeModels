# HW1 VAE report package

Open `report.pdf` for the two-page report, including references. It uses the
official NeurIPS 2026 preprint style; edit `source/report.tex` to revise it.
The author line currently names the course and homework, so add your name before
submission if required. The report links to the
[GitHub repository](https://github.com/shuoqihuang07/ECE285DeepGenerativeModels)
and identifies `HW1_VAE/` as the folder for this homework.

## Contents

- `source/`: report text, the unmodified `neurips_2026.sty`, and measured values
  inserted through `results.tex`.
- `figures/`: training curves, generated samples, real test examples, and
  posterior-mean reconstructions. The report uses `training_curves.png`,
  `generated_grid.png`, and `real_test_grid.png`.
- `data/history.csv`, `config.json`, `run_summary.json`: the recorded losses for
  all 50 epochs, training settings, split counts, selected epoch, and checkpoint hashes.
- `data/generated/`: the 624 generated PNGs evaluated in the report.
- `data/real_test/`: the 624 held-out evaluation inputs, converted to grayscale
  and resized to 128x128. The original full-resolution Kaggle images remain in
  the homework's dataset directory.
- `data/evaluation_manifest.csv`: source test filenames and SHA-256 hashes of
  original files and both sets of evaluation PNGs.
- `data/dataset_manifest.csv`: all 5,856 source filenames and split/class labels.
- `data/inception_outputs.npz`: saved 2,048-D features and 1,000-way probabilities
  for both image sets. These let you recompute the scores offline.
- `data/feature_metadata.json`, `metrics.json`: preprocessing, evaluation software
  versions, pretrained weight hash, scores, and individual IS split results.
- `checkpoints/best.pt`: the epoch-45 checkpoint used for generation. Optimizer
  tensors were omitted to save space; the weights, configuration, epoch, validation
  loss, and training settings are preserved.
- `code/`: a snapshot of the model, training, generation, plotting, and tests,
  plus report asset preparation and Ignite evaluation code.
- `checksums.sha256`: hashes of all packaged deliverables except this checksum file.

The saved evaluation inputs and features are enough to rebuild the report and
recompute its scores. To repeat training, use the original Kaggle dataset.
Inception weights are downloaded when feature extraction is run for the first time.

## Setup

From this `report/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` records the versions used for evaluation. Use
`requirements.txt` to install compatible packages on your platform. Building the
PDF also requires Tectonic or a LaTeX installation.

## Recompute scores offline

```bash
python code/evaluate.py --recompute
```

This feeds the saved arrays to Ignite's `FID` and `InceptionScore` and writes
the results to `data/metrics.json`. It works offline and skips image inference.
The matrix square root in FID can still take a minute.

## Re-extract features from the exact saved images

```bash
python code/evaluate.py --device cpu --batch-size 16
```

This command starts from the bundled 128x128 8-bit PNGs. Each image becomes
three identical RGB channels, is resized to 299x299 with PIL bilinear
interpolation, and is normalized using ImageNet mean and standard deviation.
Torchvision downloads pretrained Inception-v3 weights if needed. The model's
built-in `transform_input=True` conversion is also active. A forward pass
collects both average-pool features and class probabilities.

The evaluator follows the [assigned Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)
using `Engine`, `FID`, `InceptionScore`, and PIL interpolation. For this project,
it evaluates held-out chest X-rays, applies the same normalization to both image
sets, keeps the final partial batch, and saves the extracted features.
The report's FID explicitly uses **2,048-D pooled features**. Compare its value
only with scores obtained using the same preprocessing, features, and sample
count; it differs from TensorFlow-compatible `pytorch-fid` and Clean-FID.

The installed Ignite default instead uses 1,000-way classifier probabilities.
For reference, the script also saves `fid_probability1000` using those outputs
and the same normalized inputs. That diagnostic has a different feature space
from the report's FID. Its preprocessing also differs from the older GAN tutorial.

The main IS is computed over all 624 generated images. The script also reports
the mean and standard deviation across ten splits: four with 63 images and six
with 62. That SD is not a confidence interval. Real-test IS gives some context,
but both scores rely on ImageNet features and classes, so neither establishes
medical accuracy.

## Rebuild figures and PDF

```bash
python code/build_report_assets.py
cd source
tectonic --outdir .. report.tex
```

You can also run `pdflatex -output-directory=.. report.tex` twice from `source/`,
or upload `source/` and `figures/` to Overleaf with their relative paths intact.
The [NeurIPS 2026 template](https://neurips.cc/Conferences/2026/CallForPapers)
sets the report's margins and body font sizes.

## Generate additional images from the packaged checkpoint

```bash
python code/generate.py --checkpoint checkpoints/best.pt --num-samples 64 --seed 2026 --output-dir additional_samples --device cpu
```

To reproduce the report's 624-image sequence, use `--num-samples 624`,
`--batch-size 32`, `--seed 2026`, and `--device cpu`. Keep the batch size as well
as the seed: changing it can change the sequence of random draws.

If you need to recreate the input bundle, run `python code/prepare_assets.py`.
It reads `runs/vae/best.pt` and `data/chest_xray` from the parent homework folder
and overwrites the corresponding packaged assets.

To train again, download the [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and run `code/train.py`, pointing `--data-dir` to the folder containing its
train/val/test splits. Use `data/config.json` for the original hyperparameters.
The checkpoint records automatic device selection but not the device actually
chosen or the original training library versions. The recorded software versions
describe evaluation, not the earlier training environment.

## Validation

```bash
python -m unittest discover -s code -p 'test_*.py' -v
python code/verify_package.py
```

The checker verifies file hashes, image counts, feature and probability
dimensions, finite scores, and the two-page PDF. The rendered pages were also
checked for layout problems. After editing or rebuilding the package, refresh
its checksum manifest with `python code/verify_package.py --write-checksums`.
