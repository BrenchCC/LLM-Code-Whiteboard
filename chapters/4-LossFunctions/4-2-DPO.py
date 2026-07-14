import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
):
    """
    logps: (batch,) 每个值代表每条整段response的对数概率。
    """
    # 计算policy和ref各自的chosen-rejected提升概率
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # 计算loss
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits)
    return loss.mean()

import torch.nn.functional as F

def get_sequence_logps(logits, labels, response_mask):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    response_mask: (batch, seq_len)，1 表示 response token 参与计算
    """
    shift_logits = logits[:, :-1, :]  # shift_logits: (batch, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:]  # shift_labels: (batch, seq_len-1)
    shift_mask = response_mask[:, 1:]  # shift_mask: (batch, seq_len-1)

    # 转换成对数softmax概率, log_probs: (batch, seq_len-1, vocab_size)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 取出response token的对数概率, token_logps: (batch, seq_len-1, vocab_size)
    token_logps = log_probs.gather(
      dim=-1,
      index=shift_labels.unsqueeze(-1)  # (batch, seq_len-1, 1)，在vocab维度上gather
    ).squeeze(-1)
    
    # 过滤non-response token
    sequence_logps = (token_logps * shift_mask).sum(dim=-1)
    
    return sequence_logps


if __name__ == "__main__":
    # 获得policy上chosen和rejected的对数概率得分（每个response token的对数概率求和）
    policy_chosen_logps = get_sequence_logps(policy_chosen_logits, chosen_labels, chosen_mask)
    policy_rejected_logps = get_sequence_logps(policy_rejected_logits, rejected_labels, rejected_mask)
    
    # 获得ref上chosen和rejected的对数概率得分（每个response token的对数概率求和）
    with torch.no_grad():
        ref_chosen_logps = get_sequence_logps(ref_chosen_logits, chosen_labels, chosen_mask)
        ref_rejected_logps = get_sequence_logps(ref_rejected_logits, rejected_labels, rejected_mask)
    
    # 计算DPO损失
    loss = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.1
    )
