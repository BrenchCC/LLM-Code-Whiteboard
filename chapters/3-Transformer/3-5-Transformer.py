import torch
import torch.nn as nn
# 自定义模块部分
from attention import MultiHeadAttention
from norm import RMSNorm
from ffn import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), causal mask / padding mask
        """
        # Pre-LN + MHA + Residual Connection
        x = x + self.attn(self.norm1(x), mask=mask)
        # Pre-LN + FFN + Residual Connection
        x = x + self.ffn(self.norm2(x))
        return x
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # 可学习的位置编码
        
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )  # 注意力层
        
        self.norm = RMSNorm(d_model)  # 归一化层
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 输出头
        
    def forward(self, input_ids, mask=None):
        """
        input_ids: (batch, seq_len)
        mask: (seq_len, seq_len)
        """
        # 获取input_ids相关信息
        seq_len = input_ids.shape[-1]
        device = input_ids.device
        
        # 词嵌入并添加可学习位置编码
        pos = torch.arange(seq_len, device=device)  # pos: (seq_len,)
        pos = pos.unsqueeze(0)  # pos: (1, seq_len), broadcast到batch维度
        x = self.token_emb(input_ids) + self.pos_emb(pos)  # x: (batch, seq_len, d_model)
        
        # 注意力层处理，处理过程中始终保持x: (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.norm(x)  # 归一化层
        logits = self.lm_head(x)  # 输出头, logits: (batch, seq_len, vocab_size)
        
        return logits

import torch
import torch.nn.functional as F

def top_k_sampling(logits, k=50, temperature=1.0):
    """
    logits: (batch, vocab_size)
    return: (batch, 1)
    """
    if temperature <= 0:  # temperature=0即贪心搜索
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # 温度+top-k采样, topk_logits/idx: (batch, k)
    logits = logits / temperature
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    
    # 只对top-k进行softmax归一化，并取下标映射回原始id
    probs = F.softmax(topk_logits, dim=-1)  # probs: (batch, k)
    sampled_idx = torch.multinomial(probs, num_samples=1)  # sampled_idx: (batch, 1)
    next_token = torch.gather(topk_idx, dim=-1, index=sampled_idx)  # next_token: (batch, 1)
    return next_token

if __name__ == "__main__":
    batch = 2
    seq_len = 8
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_len = 1024
    
    model = DecoderOnlyTransformer(
      vocab_size=vocab_size,
      d_model=d_model,
      num_heads=num_heads,
      d_ff=d_ff,
      num_layers=num_layers,
      max_seq_len=max_seq_len
    )
    
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    
    mask = torch.triu(
      torch.ones(seq_len, seq_len, dtype=torch.bool),
      diagonal=1
    )
    
    logits = model(input_ids, mask=mask)  # logits: (batch, seq_len, vocab_size)
    
    next_token_id = top_k_sampling(logits, k=50, temperature=0.8)  # next_token_id: (batch,)

def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
