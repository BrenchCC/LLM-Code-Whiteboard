import torch

def mask_mean(loss, mask=None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()
    
def grpo_loss(
    new_logprobs,
    old_logprobs,
    rewards,
    group_size,
    mask=None,
    ref_logprobs=None,
    clip_eps=0.2,
    kl_coef=0.1,
    eps=1e-8
):
    """
    new_logprobs: (batch, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch, seq_len) 每时间步下旧策略模型输出的对数概率
    rewards: (batch,) 和PPO不同，PPO中returns是逐token的，GRPO中rewards对整个sequence计算
    group_size: GRPO中一组的大小
    mask: (batch, seq_len)
    ref_logprobs: (batch, seq_len) 每时间步下参考模型输出的对数概率
    clip_eps是clip中的eps，eps是计算组间相对优势时标准差用到的eps
    返回的是整个batch一起算出的loss
    """
    # 获取组数
    batch = new_logprobs.shape[0]
    assert batch % group_size == 0
    num_group = batch // group_size
    
    # 1. 计算相对优势
    group_rewards = rewards.view(num_group, group_size)  # group_rewards: (num_group, group_size)
    group_mean = group_rewards.mean(dim=-1, keepdim=True)  # group_mean: (num_group, 1)
    group_std = group_rewards.std(dim=-1, keepdim=True, unbiased=False)  # group_std: (num_group, 1)
    advantages = (group_rewards - group_mean) / (group_std + eps)  # advantages: (num_group, group_size)
    advantages = advantages.view(batch, 1)  # advantages: (batch, 1)
    
    # 2. policy_loss
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, min=1.0-clip_eps, max=1.0+clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)
    
    # 3. kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0
        
    # 4. total loss
    loss = policy_loss + kl_coef * kl_loss
    return loss
