import numpy as np


class GaussianThompsonSampler:
    """
    Simple Gaussian Thompson Sampling.
    Each asset is modeled with a Normal distribution over returns.
    Variance is assumed fixed to keep the model stable and interpretable.
    """

    def __init__(self, assets, prior_mean=0.0, prior_std=0.02):
        self.assets = assets
        self.means = {a: prior_mean for a in assets}
        self.counts = {a: 0 for a in assets}
        self.prior_std = prior_std

    def sample(self):
        samples = {}
        for asset in self.assets:
            std = self.prior_std / np.sqrt(self.counts[asset] + 1)
            samples[asset] = np.random.normal(self.means[asset], std)
        return samples

    def select_asset(self):
        samples = self.sample()
        return max(samples, key=samples.get)

    def update(self, asset, reward):
        n = self.counts[asset]
        current_mean = self.means[asset]

        # Incremental mean update
        new_mean = (current_mean * n + reward) / (n + 1)

        self.means[asset] = new_mean
        self.counts[asset] += 1