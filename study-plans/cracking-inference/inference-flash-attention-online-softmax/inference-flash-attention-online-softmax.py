import torch

def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor shaped (batch, query length, value width).
    """
    # query (B, Tq, Dk)
    # key (B, Tk, Dk)
    # value (B, Tk, Dv)
    B, Tq, Dk = query.shape
    Tk = key.shape[1]
    Dv = value.shape[-1]

    m = torch.full((B, Tq), -torch.inf)
    l = torch.zeros((B, Tq))
    O = torch.zeros((B, Tq, Dv))

    for i in range(0, Tq, query_block_size):
        q_start = i
        q_end = min(i + query_block_size, Tq)
        Q = query[:, q_start:q_end, :] # (B, Bq, Dk)
        for j in range(0, Tk, key_block_size):
            k_start = j
            k_end = min(j + key_block_size, Tk)
            K = key[:, k_start:k_end, :] # (B, Bk, Dk)
            V = value[:, k_start:k_end, :] # (B, Bk, Dv)

            scores = Q @ K.transpose(-2, -1) / Dk ** 0.5 # (B, Bq, Bk)
            mask = None
            if causal:
                tq = torch.arange(q_start, q_end) # (Bq, )
                tk = torch.arange(k_start, k_end) # (Bk, )
                mask = tq[:, None] < tk[None, :] # (Bq, Bk)
                scores = scores.masked_fill(mask, -torch.inf)
                
            max_values = torch.max(scores, dim=-1).values # (B, Bq)

            m_old = m[:, q_start:q_end] # (B, Bq)
            l_old = l[:, q_start:q_end] # (B, Bq)

            m_new = torch.maximum(m_old, max_values) # (B, Bq) 
            if mask is not None:
                m_new = torch.where(m_new == -torch.inf, m_old, m_new)
            correction = torch.exp(m_old - m_new) # (B, Bq)
            l_new = l_old * correction + torch.sum(torch.exp(scores - m_new[..., None]), dim=-1)

            O_old = O[:, q_start:q_end, :] # (B, Bq, Dv)
            O[:, q_start:q_end, :] = torch.exp(scores - m_new[..., None]) @ V + O_old * correction[..., None]

            l[:, q_start:q_end] = l_new
            m[:, q_start:q_end] = m_new

    return O / l[..., None]
