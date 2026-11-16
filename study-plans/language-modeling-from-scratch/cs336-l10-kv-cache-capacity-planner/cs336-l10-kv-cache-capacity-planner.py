def kv_cache_capacity_planner(
    model_parameters: int, bytes_per_parameter: int, num_layers: int,
    sequence_length: int, num_attention_heads: int, num_kv_heads: int,
    head_dim: int, bytes_per_element: int, memory_per_gpu: int,
    reserved_memory: int, memory_bandwidth: int,
) -> dict:
    """
    Returns a dict: model_bytes, kv_bytes_per_request, max_batch_size, seconds_per_token, tokens_per_second.
    """
    model_bytes = model_parameters * bytes_per_parameter
    kv_bytes_per_request = 2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element

    max_batch_size = max(0, (memory_per_gpu - reserved_memory - model_bytes)//kv_bytes_per_request)
    seconds_per_token = (model_bytes + max_batch_size * kv_bytes_per_request) / memory_bandwidth
    tokens_per_second = max_batch_size / seconds_per_token

    return {
        "model_bytes": model_bytes,
        "kv_bytes_per_request": kv_bytes_per_request,
        "max_batch_size": max_batch_size,
        "seconds_per_token": seconds_per_token,
        "tokens_per_second": tokens_per_second,
    }
