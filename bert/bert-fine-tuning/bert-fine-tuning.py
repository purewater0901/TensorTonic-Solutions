import numpy as np

def softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shifted_logit = logits - max_logit
    exp_ = np.exp(shifted_logit)
    return exp_ / np.sum(exp_, axis=-1, keepdims=True)

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    # W (D, C)
    # b (C,)
    B = hidden_states.shape[0]
    C = classifier_b.shape[0]
    hidden_state = hidden_states[:, 0, :] # (B, D)
    logits = hidden_state @ classifier_W + classifier_b

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )
    p = np.exp(log_probs)

    loss = -np.mean(log_probs[np.arange(B), labels])

    grad_logits = p.copy()
    grad_logits[np.arange(B), labels] -= 1
    grad_logits /= B

    grad_W = hidden_state.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0)

    new_classifier_W = classifier_W - learning_rate * grad_W
    new_classifier_b = classifier_b - learning_rate * grad_b

    return {
        "new_classifier_W": new_classifier_W,
        "new_classifier_b": new_classifier_b,
        "loss": loss,
    }

    