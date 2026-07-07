import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w = np.zeros(X.shape[1])
        b = 0
        n = len(y)
        for _ in range(epochs):
            y_hat = X @ w + b
            loss = np.mean((y_hat - y) ** 2)
            grad_w = 2 * X.T @ (y_hat - y) / n
            grad_b = 2 * np.sum(y_hat - y) / n
            w -= lr * grad_w
            b -= lr * grad_b
        return np.round(w, 5), round(b, 5)




        
