import torch


def construct_attn_input_from_map(h, order_map: dict, cat_seq=False):
    """
    Produce the inputs for the cross-view attention layer.

    Args:
        h (torch.Tensor): The hidden state of shape: [B, N, THW, self.hidden_size],
                            where T is the number of time frames and N the number of cameras.
        order_map (dict): key for query index, values for kv indexes.
        cat_seq (bool): if True, cat kv in seq length rather than batch size.
    Returns:
        h_q (torch.Tensor): The hidden state for the target views
        h_kv (torch.Tensor): The hidden state for the neighboring views
        back_order (torch.Tensor): The camera index for each of target camera in h_q
    """
    B = len(h)
    h_q, h_kv, back_order = [], [], []

    for target, values in order_map.items():
        if cat_seq:
            h_q.append(h[:, target])
            h_kv.append(torch.cat([h[:, value] for value in values], dim=1))
            back_order += [target] * B
        else:
            for neighbor in values:
                h_q.append(h[:, target])
                h_kv.append(h[:, neighbor])
                back_order += [target] * B

    h_q = torch.cat(h_q, dim=0)
    h_kv = torch.cat(h_kv, dim=0)
    back_order = torch.LongTensor(back_order)

    return h_q, h_kv, back_order