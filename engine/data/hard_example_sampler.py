"""
OHEM Strategy 3: Image-level hard example sampling.

Tracks per-image training loss and oversamples harder images in subsequent epochs.
Inspired by Online Hard Example Mining (arXiv:1604.03540).
"""

import numpy as np
from torch.utils.data import Sampler


class HardExampleSampler(Sampler):
    """Weighted sampler that oversamples images with higher training loss.

    After each epoch, the trainer updates this sampler with per-image losses.
    The sampler then creates a probability distribution biased toward
    harder images for the next epoch.

    Usage:
        1. Create sampler: sampler = HardExampleSampler(len(dataset))
        2. During training: sampler.update_losses(image_indices, losses)
        3. At epoch end: sampler.recompute_weights()
        4. Next epoch uses weighted sampling automatically
    """

    def __init__(self, dataset_size, beta=2.0, momentum=0.9, min_weight=0.1):
        """
        Args:
            dataset_size: total number of training images
            beta: temperature for loss-to-weight conversion (higher = more focus on hard)
            momentum: EMA momentum for loss tracking (0.9 = smooth)
            min_weight: minimum sampling weight (prevents complete ignoring of easy images)
        """
        self.dataset_size = dataset_size
        self.beta = beta
        self.momentum = momentum
        self.min_weight = min_weight

        # Initialize uniform weights
        self.loss_history = np.ones(dataset_size, dtype=np.float32)
        self.weights = np.ones(dataset_size, dtype=np.float32) / dataset_size
        self._initialized = False
        self._epoch = 0

    def set_epoch(self, epoch):
        """Set epoch for reproducibility (used by DistributedSampler convention)."""
        self._epoch = epoch

    def update_losses(self, image_indices, losses):
        """Update loss history after each batch.

        Args:
            image_indices: list/array of dataset indices in this batch
            losses: corresponding per-image loss values (list/array of floats)
        """
        for idx, loss in zip(image_indices, losses):
            if idx < 0 or idx >= self.dataset_size:
                continue
            if self._initialized:
                self.loss_history[idx] = (
                    self.momentum * self.loss_history[idx] +
                    (1 - self.momentum) * loss
                )
            else:
                self.loss_history[idx] = loss

    def recompute_weights(self):
        """Recompute sampling weights from loss history. Call at epoch end."""
        self._initialized = True
        w = np.power(self.loss_history, self.beta)
        w = np.clip(w, self.min_weight, None)
        self.weights = w / w.sum()

    def get_stats(self):
        """Return statistics about current sampling weights for logging."""
        return {
            'weight_min': float(self.weights.min()),
            'weight_max': float(self.weights.max()),
            'weight_std': float(self.weights.std()),
            'loss_mean': float(self.loss_history.mean()),
            'loss_max': float(self.loss_history.max()),
            'effective_samples': float(1.0 / (self.weights ** 2).sum()),  # effective sample size
        }

    def __iter__(self):
        rng = np.random.RandomState(self._epoch)
        indices = rng.choice(
            self.dataset_size,
            size=self.dataset_size,
            replace=True,
            p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.dataset_size
