import math
import unittest

import torch

from model import VAE, vae_loss


class VAETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = VAE(image_size=32, latent_dim=4, base_channels=4)

    def test_forward_backward_and_sample(self):
        images = torch.rand(3, 1, 32, 32)
        logits, mu, logvar = self.model(images)
        self.assertEqual(logits.shape, images.shape)
        self.assertEqual(mu.shape, (3, 4))
        self.assertEqual(logvar.shape, mu.shape)
        loss = vae_loss(logits, images, mu, logvar)["loss"]
        loss.backward()
        for head in (self.model.fc_mu, self.model.fc_logvar, self.model.fc_decode):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        samples = self.model.sample(5)
        self.assertEqual(samples.shape, (5, 1, 32, 32))
        self.assertTrue(((samples >= 0) & (samples <= 1)).all())

    def test_loss_matches_analytic_value_and_batch_scaling(self):
        logits = torch.zeros(1, 1, 2, 2)
        target = torch.ones_like(logits)
        mu = torch.tensor([[1.0, 2.0]])
        logvar = torch.zeros_like(mu)
        losses = vae_loss(logits, target, mu, logvar)
        self.assertAlmostEqual(losses["reconstruction"].item(), 4 * math.log(2), places=5)
        self.assertAlmostEqual(losses["kl"].item(), 2.5, places=5)
        self.assertAlmostEqual(losses["loss"].item(), 4 * math.log(2) + 2.5, places=5)
        repeated = vae_loss(logits.repeat(3, 1, 1, 1), target.repeat(3, 1, 1, 1),
                            mu.repeat(3, 1), logvar.repeat(3, 1))
        self.assertAlmostEqual(losses["loss"].item(), repeated["loss"].item(), places=5)
        self.assertEqual(vae_loss(logits, target, torch.zeros_like(mu), logvar)["kl"].item(), 0)
        self.assertEqual(vae_loss(logits, target, mu, logvar, beta=0)["loss"].item(),
                         losses["reconstruction"].item())

    def test_reparameterization_statistics(self):
        mu = torch.full((20000, 1), 2.0)
        logvar = torch.full_like(mu, math.log(4.0))
        z = self.model.reparameterize(mu, logvar)
        self.assertAlmostEqual(z.mean().item(), 2, delta=0.06)
        self.assertAlmostEqual(z.var().item(), 4, delta=0.15)

    def test_invalid_dimensions(self):
        for kwargs in ({"image_size": 30}, {"latent_dim": 0}, {"base_channels": 0}):
            with self.assertRaises(ValueError):
                VAE(**kwargs)


if __name__ == "__main__":
    unittest.main()
