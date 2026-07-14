import torch.nn.functional as F

if __name__ == "__main__":
    
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    
    # shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
    shift_logits = logits[:, :-1, :].contiguous() 
    # shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
    shift_labels = labels[:, 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)

import torch

def cross_entropy(logits, labels):
    """
    logits: (batch * (seq_len-1), vocab_size)
    labels: (batch * (seq_len-1),)
    """
    # 防止softmax溢出, logits: (batch * (seq_len-1), vocab_size)
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    # log_softmax: (batch * (seq_len-1), vocab_size)
    exp_logits = torch.exp(logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    log_probs = torch.log(probs)
    # 取对应标签id对应位置的负对数
    n = labels.size(0)
    loss = -log_probs[torch.arange(n, device=labels.device), labels]
    return loss.mean()


if __name__ == "__main__":
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    
    # shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
    shift_logits = logits[:, :-1, :].contiguous() 
    # shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
    shift_labels = labels[:, 1:].contiguous()
    
    loss = cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
