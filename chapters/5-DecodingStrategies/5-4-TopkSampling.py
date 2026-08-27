import torch
import torch.nn.functional as F

def top_k_sampling(logits, k = 50, temperature = 1.0):
    """
    logits: (batch, vocab_size)
    return: (batch, 1)
    """
    if temperature <= 0:  # temperature=0即贪心搜索
        return torch.argmax(logits, dim = -1, keepdim = True)
    
    # 温度+top-k采样, topk_logits/idx: (batch, k)
    logits = logits / temperature
    topk_logits, topk_idx = torch.topk(logits, k, dim = -1)
    
    # 只对top-k进行softmax归一化，并取下标映射回原始id
    probs = F.softmax(topk_logits, dim = -1)  # probs: (batch, k)
    sampled_idx = torch.multinomial(probs, num_samples = 1)  # sampled_idx: (batch, 1)
    next_token = torch.gather(topk_idx, dim = -1, index = sampled_idx)  # next_token: (batch, 1)
    return next_token
