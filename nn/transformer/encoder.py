import torch
import torch.nn as nn

from attention import MultiHeadAttention
from ffn import FFN
from embed import SinusoidalPositionalEncoding, TokenEmbedding

from utils import replicate


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super(EncoderBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, Q, K, V, mask=None):
        # for multi-head-attention
        attn, _ = self.multi_head_attn(Q, K, V, mask)
        # add the residual connection
        attn += V
        attn_norm = self.layer_norm(attn)

        # for feed-forward
        fc = self.ffn(attn_norm)
        fc += attn_norm
        fc_norm = self.layer_norm(fc)
        return fc_norm


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, num_heads, d_model, d_ff):
        super(Encoder, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, seq_len)

        self.encoder_block = EncoderBlock(d_model, d_ff, num_heads)
        self.encoder_blocks = replicate(self.encoder_block, N=6)

    def forward(self, x, mask=None):
        out = self.pos_embed(self.token_embed(x))
        for block in self.encoder_blocks:
            """
            Why use the same tensor for Q, K, V?
            In self-attention, the goal is for each position in the sequence to compute relationships (dependencies) with other
            positions within the same sequence. Using the same tensor for Q, K, and V ensures that:
            Q (queries) comes from the same input, so each position is "asking" questions based on itself.
            K (keys) comes from the same input, so every position "describes" itself in the context of the same sequence.
            V (values) also comes from the same input, so the information being aggregated is consistent with the rest of the sequence.
            By sharing the same input tensor, we compute self-similarities between positions in a consistent and natural way.
            """
            out = block(out, out, out, mask=mask)

        return out


if __name__ == "__main__":
    # Define the test parameters
    batch_size = 4  # Number of sequences in the batch
    seq_len = 10  # Length of each sequence
    vocab_size = 100  # Vocabulary size (number of unique tokens)
    d_model = 64  # Dimensionality of model embeddings
    d_ff = 256  # Dimensionality of feed-forward network
    num_heads = 4  # Number of attention heads
    num_blocks = 6  # Number of encoder blocks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a random input tensor of token indices (batch_size, seq_len)
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Instantiate the encoder
    encoder = Encoder(vocab_size, seq_len, num_heads, d_model, d_ff).to(device)

    # Forward pass
    output = encoder(test_input)

    # Print input and output shapes
    print("Test Input Shape:", test_input.shape)  # (batch_size, seq_len)
    print("Encoder Output Shape:", output.shape)  # (batch_size, seq_len, d_model)
