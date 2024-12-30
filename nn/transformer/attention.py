import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        # Step 1: Compute the dot product between Q and K^T
        # Q, K, V: (batch_size, seq_len, d_model)
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # Shape: (batch_size, seq_len, seq_len)

        # Step 2: Scale the dot product by sqrt(d_k)
        d_k = Q.size(-1)  # d_k = d_model
        attn_scores = attn_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Step 3: Apply the mask (optional) to the attention scores (e.g., for padding tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # Shape: (batch_size, seq_len, seq_len)
        attn_outputs = torch.matmul(attn_weights, V)

        return attn_outputs, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.d_q = self.d_k

        # Linear layers to project Q, K, V for each head
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection matrix (to project the concatenated output)
        # since d_model = d_v * h
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Step 1: Project Q, K, V using the learned linear layers
        Q = self.W_Q(Q)  # Shape: (batch_size, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # Step 2: Split Q, K, V into multiple heads
        Q = Q.view(
            batch_size, -1, self.num_heads, self.d_k
        )  # Shape: (batch_size, seq_len, num_heads, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)

        # Step 3: Transpose to get the shape (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)  # Shape: (batch_size, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 4: Apply scaled dot-product attention for each head

        output, attn_weights = self.attn(Q, K, V, mask)

        # Step 5: Concatenate the outputs of all heads
        output = output.transpose(1, 2)  # Shape: (batch_size, seq_len, num_heads, d_k)
        output = output.contiguous().view(
            batch_size, -1, self.d_model
        )  # Shape: (batch_size, seq_len, d_model)

        # Step 6: Apply the final linear layer
        output = self.W_O(output)  # Shape: (batch_size, seq_len, d_model)

        return output, attn_weights


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512  # Dimension of the model
    num_heads = 8

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Random Q, K, V tensors for demonstration (in practice, these come from the input embeddings)
    Q = torch.randn(batch_size, seq_len, d_model).to(device)  # Query
    K = torch.randn(batch_size, seq_len, d_model).to(device)  # Key
    V = torch.randn(batch_size, seq_len, d_model).to(device)  # Value

    # Optional mask to prevent attention to padding tokens (e.g., for sequence-to-sequence tasks)
    mask = None  # For simplicity, no mask is used in this example

    # Create the attention module and apply it
    attention = ScaledDotProductAttention().to(device)
    output, attn_weights = attention(Q, K, V, mask)

    # Output the results
    print(f"Output Shape: {output.shape}")  # Should be (batch_size, seq_len, d_model)
    print(
        f"Attention Weights Shape: {attn_weights.shape}"
    )  # Should be (batch_size, seq_len, seq_len)

    multi_head_attn = MultiHeadAttention(d_model, num_heads).to(device)
    output, attn_weights = multi_head_attn(Q, K, V)
    print(f"Output Shape: {output.shape}")
    print(f"Attention Weights Shape: {attn_weights.shape}")
