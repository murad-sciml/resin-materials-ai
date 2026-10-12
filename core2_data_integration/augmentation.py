"""
GAN-based synthetic data augmentation for Core Function #2.

Implemented as a small, dependency-light GAN (NumPy only — no torch/tensorflow
required) so it runs anywhere pandas/numpy already run. It is sized for the
kind of small tabular datasets typical of an experimental-materials program
(tens to low-thousands of rows, a handful of numeric properties). For much
larger datasets, swap `SimpleGAN` for a PyTorch/TensorFlow implementation
using the same `.train()` / `.generate()` interface.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def scale_to_unit(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """Min-max scale each column to [-1, 1] (matches the generator's tanh output)."""
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    scaled = 2 * (data - lo) / span - 1
    return scaled, {"lo": lo, "hi": hi, "span": span}


def unscale_from_unit(scaled: np.ndarray, params: dict) -> np.ndarray:
    return (scaled + 1) / 2 * params["span"] + params["lo"]


class SimpleGAN:
    """Minimal GAN: 1 hidden layer generator + discriminator, trained with
    vanilla (non-saturating) adversarial loss and manual backpropagation."""

    def __init__(self, n_features: int, noise_dim: int = 8, hidden_dim: int = 16,
                 lr: float = 0.01, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.noise_dim = noise_dim
        self.n_features = n_features
        self.lr = lr

        scale = 0.1
        self.Wg1 = rng.normal(0, scale, (noise_dim, hidden_dim))
        self.bg1 = np.zeros(hidden_dim)
        self.Wg2 = rng.normal(0, scale, (hidden_dim, n_features))
        self.bg2 = np.zeros(n_features)

        self.Wd1 = rng.normal(0, scale, (n_features, hidden_dim))
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.normal(0, scale, (hidden_dim, 1))
        self.bd2 = np.zeros(1)

    # -- forward passes -----------------------------------------------------
    def _generator_forward(self, z: np.ndarray):
        h1 = np.tanh(z @ self.Wg1 + self.bg1)
        out = np.tanh(h1 @ self.Wg2 + self.bg2)
        return h1, out

    def _discriminator_forward(self, x: np.ndarray):
        pre = x @ self.Wd1 + self.bd1
        h1 = _relu(pre)
        logits = h1 @ self.Wd2 + self.bd2
        prob = _sigmoid(logits)
        return h1, prob

    # -- one adversarial training step --------------------------------------
    def _train_step(self, real_batch: np.ndarray) -> tuple[float, float]:
        n = real_batch.shape[0]
        eps = 1e-8

        # ---- discriminator update ----
        z = self.rng.normal(0, 1, (n, self.noise_dim))
        _, fake = self._generator_forward(z)
        h1_real, prob_real = self._discriminator_forward(real_batch)
        h1_fake, prob_fake = self._discriminator_forward(fake)
        d_loss = -np.mean(np.log(prob_real + eps) + np.log(1 - prob_fake + eps))

        grad_logit_real = (prob_real - 1) / n
        grad_logit_fake = (prob_fake - 0) / n

        gWd2 = h1_real.T @ grad_logit_real + h1_fake.T @ grad_logit_fake
        gbd2 = grad_logit_real.sum(0) + grad_logit_fake.sum(0)

        dz1_real = (grad_logit_real @ self.Wd2.T) * _relu_grad(real_batch @ self.Wd1 + self.bd1)
        dz1_fake = (grad_logit_fake @ self.Wd2.T) * _relu_grad(fake @ self.Wd1 + self.bd1)
        gWd1 = real_batch.T @ dz1_real + fake.T @ dz1_fake
        gbd1 = dz1_real.sum(0) + dz1_fake.sum(0)

        self.Wd2 -= self.lr * gWd2
        self.bd2 -= self.lr * gbd2
        self.Wd1 -= self.lr * gWd1
        self.bd1 -= self.lr * gbd1

        # ---- generator update (non-saturating loss) ----
        z = self.rng.normal(0, 1, (n, self.noise_dim))
        gh1, fake = self._generator_forward(z)
        h1_fake, prob_fake = self._discriminator_forward(fake)
        g_loss = -np.mean(np.log(prob_fake + eps))

        grad_logit = (prob_fake - 1) / n
        dz1 = (grad_logit @ self.Wd2.T) * _relu_grad(fake @ self.Wd1 + self.bd1)
        dfake = dz1 @ self.Wd1.T

        dout = dfake * (1 - fake ** 2)
        gWg2 = gh1.T @ dout
        gbg2 = dout.sum(0)
        dh1 = dout @ self.Wg2.T
        dz1g = dh1 * (1 - gh1 ** 2)
        gWg1 = z.T @ dz1g
        gbg1 = dz1g.sum(0)

        self.Wg2 -= self.lr * gWg2
        self.bg2 -= self.lr * gbg2
        self.Wg1 -= self.lr * gWg1
        self.bg1 -= self.lr * gbg1

        return float(d_loss), float(g_loss)

    def train(self, data_scaled: np.ndarray, epochs: int = 800, batch_size: int = 32,
               verbose_every: int = 0) -> list[tuple[float, float]]:
        n = data_scaled.shape[0]
        history = []
        for epoch in range(epochs):
            idx = self.rng.choice(n, size=min(batch_size, n), replace=n < batch_size)
            batch = data_scaled[idx]
            d_loss, g_loss = self._train_step(batch)
            history.append((d_loss, g_loss))
            if verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch:4d}  d_loss={d_loss:.4f}  g_loss={g_loss:.4f}")
        return history

    def generate(self, n_samples: int) -> np.ndarray:
        z = self.rng.normal(0, 1, (n_samples, self.noise_dim))
        _, samples = self._generator_forward(z)
        return samples


def generate_synthetic(real_data: np.ndarray, n_samples: int, epochs: int = 800,
                        seed: int = 42) -> np.ndarray:
    """Convenience wrapper: scale -> train -> generate -> unscale."""
    scaled, params = scale_to_unit(real_data)
    gan = SimpleGAN(n_features=real_data.shape[1], seed=seed)
    gan.train(scaled, epochs=epochs)
    synthetic_scaled = gan.generate(n_samples)
    return unscale_from_unit(synthetic_scaled, params)


def distribution_similarity(real: np.ndarray, synthetic: np.ndarray, feature_names: list[str] | None = None) -> dict:
    """
    Per-feature Kolmogorov-Smirnov statistic + Wasserstein distance, used for
    the 'Distribution similarity of synthetic data' performance metric.
    Lower KS statistic / Wasserstein distance = closer synthetic distribution.
    """
    n_features = real.shape[1]
    feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
    report = {}
    for i, name in enumerate(feature_names):
        ks_stat, ks_p = ks_2samp(real[:, i], synthetic[:, i])
        w_dist = wasserstein_distance(real[:, i], synthetic[:, i])
        report[name] = {"ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p),
                         "wasserstein_distance": float(w_dist)}
    report["mean_ks_statistic"] = float(np.mean([v["ks_statistic"] for v in report.values()]))
    return report
