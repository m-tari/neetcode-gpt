import math
import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, H, D_h) -> (B, H, T, D_h) so attention is over sequence, not heads
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand each KV head to serve (num_heads // num_kv_heads) query heads
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Causal mask so each token only attends to itself and prior tokens
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = nn.functional.softmax(scores, dim=-1)

        # (B, H, T, D_h) -> (B, T, H*D_h)
        output = scores @ v
        output = output.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        output = self.output_proj(output)

        return torch.round(output, decimals=4)
