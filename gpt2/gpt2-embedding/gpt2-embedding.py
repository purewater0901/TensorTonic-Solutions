import torch

def gpt2_embedding(token_ids: torch.Tensor, token_embedding: torch.Tensor, position_embedding: torch.Tensor) -> torch.Tensor:
    """
    Returns token-plus-position embeddings with shape (sequence, model_width).
    """
    # token_ids (T, )
    # token_embedding: (num_tokens, E)
    # position_embedding: # (total_sequence, E)
    T = len(token_ids)
    positions = torch.arange(T)
    tokens = token_embedding[token_ids] + position_embedding[positions]

    return tokens