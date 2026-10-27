# Chest X-ray VAE (PyTorch)

A convolutional variational autoencoder for the HW1 chest X-ray generation task.
It learns from both NORMAL and PNEUMONIA images without using their labels.

## Setup and data

Run these commands from `HW1_VAE` with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract the [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
Pass the directory that directly contains `train`, `val`, and `test`:

```text
data/chest_xray/
  train/
    NORMAL/*.jpeg
    PNEUMONIA/*.jpeg
  val/
    NORMAL/*.jpeg
    PNEUMONIA/*.jpeg
  test/
    NORMAL/*.jpeg
    PNEUMONIA/*.jpeg
```

The loader recursively reads JPEG/PNG files in each selected split, converts them
to one grayscale channel, resizes to a square, and scales intensities to [0, 1].
Training reads only `train/`; checkpoint selection reads only `val/`. The supplied
validation split is small, so validation estimates may be noisy. Keep `test/`
held out for final evaluation. Data and generated outputs are ignored by Git.

## Train

```bash
python train.py --data-dir data/chest_xray --epochs 50 --batch-size 64
```

The default model uses 128x128 images, four stride-2 convolutional encoder layers
with 32/64/128/256 channels, a 128-dimensional latent vector, and a mirrored
transposed-convolution decoder. Independent linear heads predict the Gaussian
mean and log variance. Sampling uses `z = mu + exp(0.5 * logvar) * epsilon` with
`epsilon ~ N(0, I)`; generation draws directly from the standard-normal prior.

The objective is the negative evidence lower bound:

```text
reconstruction = mean_over_batch(sum_over_pixels(BCEWithLogits(decoder(z), x)))
KL = mean_over_batch(-0.5 * sum_over_latents(1 + logvar - mu^2 - exp(logvar)))
loss = reconstruction + beta * KL
```

The default `beta=1` gives the standard VAE objective. BCE on continuous grayscale
intensities is used as a reconstruction surrogate. The decoder returns logits;
sigmoid is applied only to displayed or saved images. Losses are summed over
pixels/latent dimensions and averaged over images, so their magnitudes depend on
image resolution. Optimization uses Adam with learning rate 0.001.

CUDA, Apple MPS, or CPU is selected automatically. Use `--device cpu` to override.
Other options include `--image-size` (a multiple of 16), `--latent-dim`,
`--base-channels`, `--lr`, `--beta`, `--seed`, `--workers`, and `--output-dir`.
Reduce batch size if memory is limited. A fixed seed controls random initialization
and sampling; accelerator kernels can still vary across runs.

Outputs in `runs/vae/`:

- `best.pt`: model with the lowest validation objective; `last.pt`: latest model.
  Both include model configuration, weights, optimizer state, and training settings.
- `config.json`: hyperparameters and paths.
- `history.csv` and `loss_curves.png`: train/validation total, reconstruction, and KL losses.
- `samples_001.png`, etc.: prior samples from fixed latent vectors each epoch.
- `reconstructions.png`: validation originals on the top row and reconstructions
  from posterior means on the bottom row.

## Generate

```bash
python generate.py --checkpoint runs/vae/best.pt --num-samples 64 --output-dir generated
```

This saves individual grayscale PNGs and `grid.png`. Generation processes images
in batches (`--batch-size`) and accepts `--seed` and `--device`.

## Verify

```bash
python -m unittest -v test_model.py test_plot_history.py
```

The checks cover tensor shapes, gradients through both encoder heads and decoder,
the analytic reconstruction/KL loss, batch averaging, and sampling statistics.

To regenerate loss curves from a completed run without retraining:

```bash
python plot_history.py --history runs/vae/history.csv
```

Plotting uses Matplotlib's Agg canvas directly. This avoids loading pyplot's
style library, which can mistake macOS `._` metadata files on external drives
for text style files and raise a `UnicodeDecodeError`.

The completed [report package](report/README.md) contains the two-page NeurIPS
report, the selected checkpoint, 624 generated images, the held-out evaluation
inputs, training curves, and reproducible Ignite metric code. Measured results
are FID 283.72 (torchvision 2,048-D pooled features) and Inception Score 2.095.
See the package README for the exact preprocessing, metric limitations, and
instructions to recompute scores or rebuild the PDF without retraining.
