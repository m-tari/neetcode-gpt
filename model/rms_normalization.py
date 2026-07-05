import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        eps = 1e-5
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        rms = np.sqrt(np.mean(x**2, axis=0) + eps)
        x_hat = x / rms
        return np.round(gamma * x_hat, 4).tolist()


