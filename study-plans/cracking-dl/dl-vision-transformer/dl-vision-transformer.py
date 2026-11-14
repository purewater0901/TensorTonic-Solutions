import numpy as np

def vit_patch_embed(image: list, patch_size: int, W_proj: list, b_proj: list, W_cls: list, W_pos: list) -> dict:
    """
    Returns a dictionary of lists: patches, patch_embeddings, tokens, output.
    """
    image = np.array(image, dtype=np.float64)
    W_proj = np.array(W_proj, dtype=np.float64)
    b_proj = np.array(b_proj, dtype=np.float64)
    W_cls = np.array(W_cls, dtype=np.float64)
    W_pos = np.array(W_pos, dtype=np.float64)
    
    C, H, W = image.shape
    h_size = H // patch_size
    w_size = W // patch_size

    patches = []
    for h in range(h_size):
        for w in range(w_size):
            patch = image[:, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
            patches.append(patch.flatten()[None, :])

    patches = np.concatenate(patches, axis=0) # (N, C*p*p)
    patch_embeddings = patches @ W_proj.T + b_proj

    tokens = np.concatenate([W_cls, patch_embeddings], axis=0) # (N+1, C*p*p)
    output = tokens + W_pos[:len(tokens)]

    return {
        "patches": np.round(patches,4).tolist(),
        "patch_embeddings": np.round(patch_embeddings,4).tolist(),
        "tokens": np.round(tokens,4).tolist(),
        "output": np.round(output,4).tolist(),
    }

            




            
