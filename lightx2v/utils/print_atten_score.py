import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention

    Args:
        Q: Query tensor [batch_size, num_heads, seq_len, d_k]
        K: Key tensor [batch_size, num_heads, seq_len, d_k]
        V: Value tensor [batch_size, num_heads, seq_len, d_k]
        mask: Attention mask (0 indicates positions to mask, 1 indicates positions to keep)

    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask_value = torch.where(mask == 0, torch.tensor(-float("inf")), torch.tensor(0.0))
        scores = scores + mask_value

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V)
    return output, scores, attention_weights


def draw_matrix(weights, save_path):
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def get_qkv_subset(x, head_index, token_start, token_end):
    """
    x : [seq_len, num_heads, head_dim]

    return: [batch_size, num_heads, seq_len, head_dim]
    batch_size = 1, num_heads = 1, seq_len = token_end - token_start
    """
    x = x[token_start:token_end, head_index, :]  # [seq_len, head_dim]
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    return x


def draw_attention_weights(q, k, v, head_index, token_start, token_end, save_path):
    """
    q k v : [seq_len, num_heads, head_dim]
    """
    q_vis = get_qkv_subset(q, head_index=head_index, token_start=token_start, token_end=token_end)
    k_vis = get_qkv_subset(k, head_index=head_index, token_start=token_start, token_end=token_end)
    v_vis = get_qkv_subset(v, head_index=head_index, token_start=token_start, token_end=token_end)
    output, scores, attention_weights = scaled_dot_product_attention(q_vis, k_vis, v_vis, mask=None)
    draw_matrix(scores[0][0].float().cpu().numpy(), save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    seq_len = 10
    num_heads = 4
    head_dim = 8

    q = torch.randn(seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)

    draw_attention_weights(q, k, v, head_index=0, token_start=0, token_end=10, save_path="scores.png")
