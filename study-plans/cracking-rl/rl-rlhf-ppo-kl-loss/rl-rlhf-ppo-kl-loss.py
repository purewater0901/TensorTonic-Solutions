import math
import numpy as np

def rlhf_ppo_kl_loss(log_probs_new: list, log_probs_old: list, log_probs_ref: list, advantages: list, clip_eps: float, kl_coef: float) -> float:
    """
    Returns the combined PPO and reference-penalty loss, rounded to four decimals.
    """
    log_probs_new = np.array(log_probs_new, dtype=np.float64)
    log_probs_old = np.array(log_probs_old, dtype=np.float64)
    log_probs_ref = np.array(log_probs_ref, dtype=np.float64)
    advantages = np.array(advantages, dtype=np.float64)

    T = len(log_probs_new)

    r = np.exp(log_probs_new - log_probs_old)
    loss_clip = -np.mean(np.minimum(r * advantages, np.clip(r, 1-clip_eps, 1+clip_eps)*advantages))

    K = np.mean(log_probs_new - log_probs_ref)

    return loss_clip + kl_coef * K