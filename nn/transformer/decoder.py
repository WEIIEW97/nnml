import torch
import torch.nn as nn

from attention import MultiHeadAttention
from ffn import FFN
from embed import SinusoidalPositionalEncoding, TokenEmbedding

from utils import replicate


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super(DecoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x_cross, x_self, cross_mask, self_mask):
        self_attn_out, _ = self.self_attn(x_self, x_self, x_self, self_mask)
        x1 = self.layer_norm(x_self + self_attn_out)

        cross_attn_out, _ = self.cross_attn(x1, x_cross, x_cross, cross_mask)
        x2 = self.layer_norm(x1 + cross_attn_out)

        ffn_out = self.ffn(x2)
        x3 = self.layer_norm(x2 + ffn_out)

        return x3


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, num_heads, d_model, d_ff):
        super(Decoder, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, seq_len)
        self.decoder_block = DecoderBlock(d_model, d_ff, num_heads)
        self.decoder_blocks = replicate(self.decoder_block, N=6)

    def forward(self, decoder_in, encoder_out, encoder_mask, decoder_mask):
        x_d = self.pos_embed(self.token_embed(decoder_in))
        for block in self.decoder_blocks:
            x = block(encoder_out, x_d, encoder_mask, decoder_mask)

        return x


if __name__ == "__main__":
    # Define parameters
    batch_size = 4  # Number of sequences in the batch
    src_len = 12  # Length of the source sequence
    tgt_len = 10  # Length of the target sequence
    vocab_size = 100  # Vocabulary size for the embeddings
    d_model = 64  # Dimensionality of model embeddings
    seq_len = max(src_len, tgt_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate dummy target sequence input for decoder
    decoder_in = torch.randint(0, vocab_size, (batch_size, tgt_len)).to(
        device
    )  # (batch_size, tgt_len)

    # Generate dummy encoder output (assume it comes from the encoder)
    encoder_out = torch.rand(batch_size, src_len, d_model).to(
        device
    )  # (batch_size, src_len, d_model)

    # Generate encoder mask (optional, for padded tokens in encoder output)
    encoder_mask = torch.ones(batch_size, src_len).bool()  # No padding in this case
    encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2).to(device)

    # Generate causal decoder mask for target sequence
    decoder_mask = torch.triu(
        torch.ones(tgt_len, tgt_len), diagonal=1
    ).bool()  # Upper triangular causal mask
    decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1).to(device)

    # Print test inputs
    print("Decoder Input Shape:", decoder_in.shape)
    print("Encoder Output Shape:", encoder_out.shape)
    print("Encoder Mask Shape:", encoder_mask.shape)
    print("Decoder Mask Shape:", decoder_mask.shape)

    # Instantiate the decoder
    decoder = Decoder(
        vocab_size=vocab_size, seq_len=seq_len, num_heads=4, d_model=d_model, d_ff=256
    ).to(device)

    # Forward pass through the decoder
    output = decoder(decoder_in, encoder_out, encoder_mask, decoder_mask)

    # Print the decoder output shape
    print(
        "Decoder Output Shape:", output.shape
    )  # Expected: (batch_size, tgt_len, d_model)
