import torch

def mask_mean(loss, mask = None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()

def ppo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    values,
    returns,
    mask = None,
    ref_logprobs = None,
    clip_eps = 0.2,
    value_coef = 0.5,
    kl_coef = 0.1
):
    """
    new_logprobs: (batch, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch, seq_len) 每时间步下旧策略模型输出的对数概率
    advantages: (batch, seq_len) 每时间步下优势估计
    values: (batch, seq_len) 每时间步下价值模型输出的预测价值
    returns: (batch, seq_len)  每时间步下的GAE回报
    mask: (batch, seq_len), 1 表示有效 token
    ref_logprobs: (batch, seq_len), 每时间步下参考模型输出的对数概率
    返回的是整个batch一起算出的loss
    """
    # 1.policy_loss
    ratio = torch.exp(new_logprobs - old_logprobs)  # 新旧策略概率比r_t，去对数处理
    
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, min = 1.0-clip_eps, max = 1.0+clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)
    
    # 2.value_loss
    value_loss = (values - returns) ** 2
    value_loss = mask_mean(value_loss, mask)

    # 3.kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0
        
    # 4.total loss
    loss = policy_loss + value_coef * value_loss + kl_coef * kl_loss
    
    return loss

import torch

def advantage_estimate(
    rewards,
    values,
    dones,
    gamma = 0.99,
    lam = 0.95
):
    """
    rewards/values: (batch, seq_len)
    dones: (batch, seq_len), 1表示该轨迹结束，后续不再算values，0表示未结束
    gamma是折扣因子，lam是GAE平滑系数
    """
    # 获取rewards相关信息并初始化
    batch, seq_len = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)  # advantages: (batch, seq_len)
    last_gae_lam = torch.zeros(batch, device = device)  # last_gae_lam: (batch,)
    
    #
    for t in reversed(range(seq_len)):
        # 计算下一时刻起的values: V(s_{t+1})
        if t == seq_len - 1:
            next_values = torch.zeros(batch, device = device)
        else:
            next_values = values[:, t+1]
        next_non_terminal = 1.0 - dones[:, t]  # 是否达到该轨迹末端
        
        # TD ERROR: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values * next_non_terminal - values[:, t]
        
        # 计算GAE
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
        advantages[:, t] = last_gae_lam
        
    # 返回优势和折扣回报,advantages: (batch, seq_len), returns:(batch, seq_len)
    returns = advantages + values
    return advantages, returns
