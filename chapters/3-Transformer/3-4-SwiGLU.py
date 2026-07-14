import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)  # gate_proj
        self.W3 = nn.Linear(d_model, d_ff, bias=False)  # up_proj
        self.W2 = nn.Linear(d_ff, d_model, bias=False)  # down_proj
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 计算gate分支和up分支
        gate = F.silu(self.W1(x))  # gate: (batch, seq_len, d_ff)
        value = self.W3(x)  # value: (batch, seq_len, d_ff)
        
        out = gate * value  # 逐元素相乘
        out = self.W2(out)
        return out
