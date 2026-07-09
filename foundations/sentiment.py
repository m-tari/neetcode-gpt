import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # 1. Embed token IDs: (B, T) -> (B, T, 16)
        # T equals the word count of the longest sentence.
        # 2. Mean-pool over T (each sentence, at dim=1) for a fixed-size bag-of-words vector -> (B, 16)
        # 3. Linear + sigmoid -> (B, 1) sentiment probability; round to 4 decimals
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
