import numpy as np
import math

def ppo_clipped_loss(log_probs_new: list, log_probs_old: list, advantages: list, clip_eps: float) -> float:
    """
    Returns the PPO clipped loss as a float rounded to four decimals.
    """
    log_probs_new = np.array(log_probs_new, dtype=np.float64)
    log_probs_old = np.array(log_probs_old, dtype=np.float64)
    advantages = np.array(advantages, dtype=np.float64)
    r_t = np.exp(log_probs_new - log_probs_old) # (T, )
    loss = -np.mean(np.minimum(r_t * advantages, np.clip(r_t, 1.0 - clip_eps, 1.0 + clip_eps) * advantages))
    return loss