# Chest X-ray VAE (PyTorch)

This folder contains the code for HW1: training a variational autoencoder to
generate chest X-rays. The model learns from NORMAL and PNEUMONIA images together,
without using their labels. It reconstructs images during training, then generates
new ones by decoding samples from a standard normal distribution.

## Setup and data

Use Python 3.10 or newer. Run the commands below from `HW1_VAE/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract the [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
Use the directory that directly contains `train/`, `val/`, and `test/`:

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

The loader reads JPEG and PNG files recursively, converts them to grayscale,
resizes them to a square, and scales pixel values to [0, 1]. Only `train/` updates
the model's weights. Validation loss on `val/` determines which checkpoint to
keep; `test/` is reserved for final evaluation. The supplied validation set has
just 16 images, so small changes in its loss need care when interpreting results.
Git ignores the downloaded data and generated outputs.

## Train

```bash
python train.py --data-dir data/chest_xray --epochs 50 --batch-size 64
```

By default, the encoder takes 128x128 images through four stride-2 convolutional
layers with 32, 64, 128, and 256 channels. Two linear heads predict the mean and
log variance of a 128-dimensional Gaussian. The decoder maps a sampled latent
vector back to an image through a linear projection and transposed convolutions.

During training, the latent sample is
`z = mu + exp(0.5 * logvar) * epsilon`, where `epsilon ~ N(0, I)`. For new images,
the decoder receives a draw directly from `N(0, I)`.

The loss balances reconstruction against KL divergence:

```text
reconstruction = mean_over_batch(sum_over_pixels(BCEWithLogits(decoder(z), x)))
KL = mean_over_batch(-0.5 * sum_over_latents(1 + logvar - mu^2 - exp(logvar)))
loss = reconstruction + beta * KL
```

With `beta=1`, this has the standard negative-ELBO form. Since the targets are
continuous grayscale intensities, BCE is used as a reconstruction surrogate.
The loss takes decoder logits directly; sigmoid is applied when displaying or
saving images. Reconstruction is summed over pixels and KL over latent
dimensions, then both are averaged over images. This means image resolution
affects the loss scale. Adam uses a default learning rate of 0.001.

The script selects CUDA, Apple MPS, or CPU automatically. Set `--device cpu` to
choose CPU explicitly, or reduce `--batch-size` if memory is limited. You can also
change `--image-size` (a multiple of 16), `--latent-dim`, `--base-channels`, `--lr`,
`--beta`, `--seed`, `--workers`, and `--output-dir`. A fixed seed helps reproduce
initialization and sampling, although results can still vary by device and library version.

Outputs in `runs/vae/`:

- `best.pt`: checkpoint with the lowest validation loss. `last.pt` holds the most
  recent epoch. Both save the configuration, weights, optimizer state, and settings.
- `config.json`: hyperparameters and paths.
- `history.csv` and `loss_curves.png`: train/validation total, reconstruction, and KL losses.
- `samples_001.png`, etc.: prior samples from fixed latent vectors each epoch.
- `reconstructions.png`: original validation images above reconstructions made
  by decoding their posterior-mean latent vectors.

## Generate

```bash
python generate.py --checkpoint runs/vae/best.pt --num-samples 64 --output-dir generated
```

The script loads the checkpoint and saves individual grayscale images plus
`grid.png`. Use `--batch-size`, `--seed`, and `--device` to control the sampling run.

## Verify

```bash
python -m unittest -v test_model.py test_plot_history.py
```

These tests check tensor shapes, gradients through both encoder heads and the
decoder projection, reconstruction/KL arithmetic, batch averaging, and sampling
statistics. They verify the implementation; they do not measure image realism.

To regenerate loss curves from a completed run without retraining:

```bash
python plot_history.py --history runs/vae/history.csv
```

Plotting uses Matplotlib's Agg canvas directly. This avoids a problem encountered
on the external drive: pyplot treated binary macOS `._` metadata as text style
files and raised a `UnicodeDecodeError` after training had finished.

The [report package](report/README.md) includes the two-page NeurIPS report,
selected checkpoint, 624 generated images, held-out evaluation inputs, curves,
and Ignite evaluation code. This run produced FID 283.72 using torchvision's
2,048-D pooled features and Inception Score 2.095. Its README explains the
preprocessing and limitations, and how to reproduce the scores or rebuild the
report from saved results.
