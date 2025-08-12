import torch
import torch.nn as nn

from attention import MultiHeadAttention


def shift_right(x: torch.Tensor) -> torch.Tensor:
    """This method shifts ith row of a matrix by i columns.

    If the input is [[1, 2 ,3], [4, 5 ,6], [7, 8, 9]] , the shifted
    result would be [[1, 2 ,3], [0, 4, 5], [6, 0, 7]] . Ideally we
    should mask out the lower triangle but it's ok for our purpose.
    """
    # Assuming x has shape (batch_size, seq_len, d_model)
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)
    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(RelativeMultiHeadAttention, self).__init_()
        self.P = 2**12
        self.key_pos_emb = nn.Parameter(
            torch.zeros((self.P * 2, num_heads, self.d_k)), requires_grad=True
        )
        self.key_pos_bias = nn.Parameter(
            torch.zeros((self.P * 2, num_heads)), requires_grad=True
        )
        self.query_pos_bias = nn.Parameter(
            torch.zeros((self.P * 2, self.d_k)), requires_grad=True
        )

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        key_pos_emb = self.key_pos_emb[self.P - k.shape[0] : self.P + q.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - k.shape[0] : self.P + q.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]
        ac = torch.einsum("ibhd,jbhd->ijbh", q + query_pos_bias, k)
        b = torch.einsum("ibhd,jhd->ijbh", q, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = shift_right(b + d)

        bd = bd[:, -k.shape[0] :]
        return ac + bd
