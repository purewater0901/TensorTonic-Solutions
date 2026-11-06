import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    # query (B, Hq, D)
    key = torch.cat([key_cache, new_key[:, :, None, :]], dim=2) # (B, Hkv, T+1, D)
    value = torch.cat([value_cache, new_value[:, :, None, :]], dim=2) # (B, Hkv, T+1, D)

    B = query.shape[0]
    D = query.shape[-1]
    G = num_query_heads // num_kv_heads

    query = query.view(B, num_kv_heads, G, D)
    scores = query @ key.transpose(2, 3) / D ** 0.5 # (B, Hkv, G, T+1)
    weights = torch.softmax(scores, dim=-1) # (B, Hkv, G, T+1)
    weights = weights.unsqueeze(3) # (B, Hkv, G, 1, T+1)
    broad_value = value.unsqueeze(2) # (B, Hkv, 1, T+1, D)
    outputs = weights @ broad_value # (B, Hkv, G, 1, D)
    outputs = outputs.unsqueeze(-2) # (B, Hkv, G, D)
    outputs = outputs.reshape(B, -1, D) # (B, Hq, D)

    return {"output": outputs, "new_key_cache": key, "new_value_cache": value}
