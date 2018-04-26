import numpy as np

class DiagonalGaussian(object):
    def __init__(self, dim):
        self._dim = dim

    def kl(self, old_means, old_log_stds, new_means, new_log_stds):
        old_stds = np.exp(old_log_stds)
        new_stds = np.exp(new_log_stds)

        numerator = np.square(old_means - new_means) + np.square(old_stds) - np.square(new_stds)
        denominator = 2 * np.square(new_stds) + 1e-8
        return np.sum(numerator/denominator + new_log_stds - old_log_stds, axis=-1)

    def sample(self, means, log_stds):
        rnd = np.random.normal(size=means.shape)
        return rnd*np.exp(log_stds) + means

