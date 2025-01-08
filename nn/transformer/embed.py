import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """Embedding class to convert a word into embedding space
        (numerical representation) :param vocab_size: the vocabulary
        size :param embed_dim: the embedding dimension.

        example: if we have 1000 vocabulary size and our embedding is 512,
        then the embedding layer will be 1000x512

        suppose we have a batch size of 64 and sequence of 15 words,
        then the output will be 64x15x512
        """
        super(TokenEmbedding, self).__init__()
        self.embed_dim = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """Forward pass :param x: the word or sequence of words :return:

        the numerical representation of the input.
        """
        return self.embed(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        """Sinusoidal Positional Embedding or Positional Encoding The
        general idea here is to add positional encoding to the input
        embedding before feeding the input vectors to the first
        encoder/decoder The positional embedding must have the same
        embedding dimension as in the embedding vectors For the
        positional encoding we use sin and cos For more details, check
        "Positional Encoding" section in the "Attention Is All You Need"
        paper.

        :param embed_dim: the size of the embedding, this must be the
            same as in embedding vector
        :param max_seq_len: the maximum sequence length (max sequence of
            words)
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # Applying sine to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        # Applying cosine to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding a batch dimension and registering it as a buffer (not a model parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x is the input token embeddings with shape (batch_size, seq_length, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


if __name__ == "__main__":
    d_model = 512
    seq_length = 10
    vocab_size = 10000
    batch_size = 3

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    token_embedding = TokenEmbedding(vocab_size, d_model).to(device)
    pos_embedding = SinusoidalPositionalEncoding(d_model, seq_length).to(device)

    # Example input: a batch of token indices (size batch_size x seq_length)
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

    # Get token embeddings
    embedded_tokens = token_embedding(
        input_tokens
    )  # Shape: (batch_size, seq_length, d_model)

    # Add positional encoding to the token embeddings
    output = pos_embedding(embedded_tokens)  # Shape: (batch_size, seq_length, d_model)

    # Print the output shape
    print(f"You are currently running on {device}")
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Token embeddings shape: {embedded_tokens.shape}")
    print(f"Output shape (after adding positional encoding): {output.shape}")
