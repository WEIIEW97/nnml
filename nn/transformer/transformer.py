import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        dst_vocab_size,
        seq_len,
        num_heads,
        d_model,
        d_ff,
        pad_token_id=0,
        device=torch.device("cuda"),
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, seq_len, num_heads, d_model, d_ff).to(
            device
        )
        self.decoder = Decoder(dst_vocab_size, seq_len, num_heads, d_model, d_ff).to(
            device
        )

        self.linear = nn.Linear(d_model, dst_vocab_size).to(device)

        self.pad_token_id = pad_token_id
        self.device = device

    def rshift(self, t):
        """Right-shifts the target sequence by one position and prepends
        the PAD or START token.

        Parameters:
        - t (Tensor): Target sequence (batch_size, tgt_len).

        Returns:
        - Tensor: Right-shifted target sequence (batch_size, tgt_len).
        """
        shift_t = torch.zeros_like(t)  # (batch_size, tgt_len)
        shift_t[:, 1:] = t[:, :-1]
        shift_t[:, 0] = self.pad_token_id
        return shift_t

    def cross_attn_square_mask(self, t):
        B, l = t.size()
        mask = torch.triu(torch.ones(l, l), diagonal=1).bool()
        return mask.expand(B, 1, l, l)

    def self_attn_square_mask(self, s):
        B, l = s.size()
        mask = torch.ones(B, l).bool()
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(self, s, t):
        self_mask = self.self_attn_square_mask(s).to(self.device)
        cross_mask = self.cross_attn_square_mask(t).to(self.device)
        enc_out = self.encoder(s, self_mask)
        t_shift = self.rshift(t)
        dec_out = self.decoder(t_shift, enc_out, self_mask, cross_mask)
        out = self.linear(dec_out)
        prob = F.softmax(out, dim=-1)
        return prob


if __name__ == "__main__":
    # Define parameters
    batch_size = 4
    src_len = 12
    tgt_len = 10
    vocab_size = 100
    seq_len = max(src_len, tgt_len)
    d_model = 64
    d_ff = 256
    num_heads = 4
    pad_token_id = 0  # Assume 0 is the padding/START token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate dummy input data
    src = torch.randint(1, vocab_size, (batch_size, src_len)).to(
        device
    )  # Source sequence
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(
        device
    )  # Target sequence

    # Instantiate the Transformer model
    transformer = Transformer(
        vocab_size,
        vocab_size,
        seq_len,
        num_heads,
        d_model,
        d_ff,
        pad_token_id,
        device=device,
    )

    # Forward pass
    output = transformer(src, tgt)

    # Print output shape
    print(
        "Transformer Output Shape:", output.shape
    )  # Expected: (batch_size, tgt_len, vocab_size)
