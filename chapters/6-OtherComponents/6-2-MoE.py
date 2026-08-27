import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        """
        x: (..., d_model)
        return: (..., d_model)
        """
        return self.net(x)
    
class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias = False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        return: 线性层输出和负载均衡损失, out: (batch, seq_len, d_model), aux_loss
        """
        # x按token展平
        batch, seq_len, d_model = x.shape
        x_flat = x.view(batch * seq_len, d_model)  # x_flat: (batch * seq_len, d_model)
        
        # 计算路由到各个专家的分数, router_logits: (batch * seq_len, num_experts)
        router_logits = self.router(x_flat)
        
        # 每个token选取top-k专家并重新归一化, topk_logits/idx/probs: (batch * seq_len, top_k)
        topk_logits, topk_idx = torch.topk(router_logits, self.top_k, dim = -1)
        topk_probs = F.softmax(topk_logits, dim = -1)
        
        # 计算负载均衡损失，moe_aux_loss之后定义
        aux_loss = moe_aux_loss(
            router_logits = router_logits, 
            topk_idx = topk_idx, 
            num_experts = self.num_experts
        )
        
        # 初始化输出, out_flat: (batch * seq_len, d_model)
        out_flat = torch.zeros_like(x_flat)
        
        # 对每个专家，处理各自的token
        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id  # 哪些token选中当前的专家
            if not mask.any():
                continue
               # 对于编号为token_idx的token，当前专家是它第which_k个top-k专家
            # token_idx和which_k是长度为selected_len的向量, selected_len为选中当前专家的token数
            token_idx, which_k = torch.where(mask)
            # 专家处理, input: (selected_len, d_model), output: (selected_len, d_model)
            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)
            # 对于每个token, 将权重回传, weight: (selected_len, 1)
            weight = topk_probs[token_idx, which_k].unsqueeze(-1)
            out_flat.index_add_(dim = 0, index = token_idx, source = expert_output * weight)
        
        # 将输出形状调整回去, out: (batch, seq_len, d_model)
        out = out_flat.view(batch, seq_len, d_model)
        return out, aux_loss

if __name__ == "__main__":
    self.fc1 = nn.Linear(d_model, d_ff)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn.functional as F

def moe_aux_loss(router_logits, topk_idx, num_experts):
    """
    router_logits: (num_tokens, num_experts)
    topk_idx: (num_tokens, top_k)
    """
    # 1.Router概率pi
    router_probs = F.softmax(router_logits, dim = -1)  # router_probs: (num_tokens, num_experts)
    pi = router_probs.mean(dim = 0)  # pi: (num_experts,)
    
    # 2.token分配比例fi
    expert_mask = F.one_hot(topk_idx, num_classes = num_experts).float()  # expert_mask: (num_tokens, top_k, num_experts)
    fi = expert_mask.mean(dim = (0, 1))  # fi: (num_experts,)
    
    aux_loss = num_experts * torch.sum(pi * fi)
    return aux_loss
