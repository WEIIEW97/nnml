import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_out=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize FFN model
    ffn_model = FFN(d_model=512, d_ff=2048).to(device)

    # Create a random tensor as an input (e.g., batch size of 2, input size of 512)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 512).to(device)

    # Pass input through FFN
    output_tensor = ffn_model(input_tensor)

    # Print the output shape
    print(f"Output Shape: {output_tensor.shape}")
    print(output_tensor)
