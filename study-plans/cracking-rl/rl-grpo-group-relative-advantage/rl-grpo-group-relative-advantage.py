import math
import numpy as np

def grpo_advantages(rewards: list, group_indices: list, eps: float = 1e-8) -> list:
    """
    Returns the group-normalized advantage list, rounded to four decimals.
    """
    rewards = np.array(rewards, dtype=np.float64)
    ids = np.unique(group_indices)

    advs = np.zeros_like(rewards)
    for id in ids:
        mask = group_indices == id
        target_rewards = rewards[mask]

        avg_reward = np.mean(target_rewards)
        std_reward = np.std(target_rewards)

        if std_reward < 1e-12:
            advs[mask] = 0.0
        else:
            advs[mask] = (target_rewards - avg_reward) / (std_reward + eps)

    return np.round(advs, 4).tolist()

        
        