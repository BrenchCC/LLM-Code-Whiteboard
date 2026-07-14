import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        assert d_model % 2 == 0
        
        self.d_model = d_model
        self.base = base
    
    def forward(self, x):
        """
        (x: batch, seq_len, d_model)
        """
        # 获取x相关信息
        seq_len = x.shape[-2]
        device = x.device
        
        # 生成频率, freq: (d_model / 2,)
        dim = torch.arange(0, self.d_model, 2, device=device)  # dim: (d_model / 2,)
        freq = self.base ** (-dim / self.d_model)
        
        # 生成旋转角, theta: (seq_len, d_model / 2)
        pos = torch.arange(seq_len, device=device)  # pos: (seq_len,)
        theta = torch.outer(pos, freq)
        
        # 计算正弦余弦值, cos/sin: (seq_len, d_model / 2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        # x拆分为奇偶维, x_odd/x_even: (batch, seq_len, d_model / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # 计算奇偶维RoPE, out_odd, out_even: (batch, seq_len, d_model / 2)
        out_even = cos * x_even - sin * x_odd
        out_odd = sin * x_even + cos * x_odd
        
        # 合并奇偶维, out: (batch, seq_len, d_model)
        out = torch.zeros_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    """
    使用上面定义好的RoPE类，此处省略
    """

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0

        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(d_model)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch维度
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # RoPE 只作用在 Q 和 K 上
        Q = self.rope(Q)
        K = self.rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        return self.o_proj(out)
