import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        # Project input through K, Q, V linear layers
        k = self.key(embedded)  # (B, T, attention_dim)
        q = self.query(embedded)  # (B, T, attention_dim)
        v = self.value(embedded)  # (B, T, attention_dim)

        # Compute attention scores
        scores = q @ k.transpose(1, 2) / math.sqrt(self.attention_dim)  # (B, T, T)

        # Apply causal mask
        T = scores.shape[1]  # context length
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        scores = torch.softmax(scores, dim=2)

        return torch.round(scores @ v, decimals=4)
