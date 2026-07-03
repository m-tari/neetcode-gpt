import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        eps = 1e-7
        losses = y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        average_loss = - np.mean(losses)
        return round(average_loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7        
        # y_true and y_pred are (n_samples, n_classes)
        # Summing across axis=1 collapses the classes into a single value per sample
        sample_losses = np.sum(y_true * np.log(y_pred + eps), axis=1)
        total_mean_loss = -np.mean(sample_losses)

        return round(total_mean_loss, 4)
        
