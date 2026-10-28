import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=128, base_channels=32):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if latent_dim < 1 or base_channels < 1:
            raise ValueError("latent_dim and base_channels must be positive")
        self.image_size = image_size
        self.latent_dim = latent_dim
        channels = [1] + [base_channels * 2**i for i in range(4)]
        self.encoder = nn.Sequential(*[
            layer
            for cin, cout in zip(channels[:-1], channels[1:])
            for layer in (nn.Conv2d(cin, cout, 4, 2, 1), nn.ReLU())
        ])
        self.feature_shape = (channels[-1], image_size // 16, image_size // 16)
        feature_dim = channels[-1] * (image_size // 16) ** 2
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feature_dim)
        decoder = []
        for cin, cout in zip(channels[:0:-1], channels[-2::-1]):
            decoder.append(nn.ConvTranspose2d(cin, cout, 4, 2, 1))
            if cout != 1:
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        features = self.encoder(x).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        features = self.fc_decode(z).reshape(-1, *self.feature_shape)
        return self.decoder(features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, num_samples):
        device = next(self.parameters()).device
        return self.decode(torch.randn(num_samples, self.latent_dim, device=device)).sigmoid()


def vae_loss(logits, target, mu, logvar, beta=1.0):
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return {"loss": reconstruction + beta * kl, "reconstruction": reconstruction, "kl": kl}
