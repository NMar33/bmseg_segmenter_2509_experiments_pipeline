# src/models/adapter.py

"""
A learnable 1x1 convolution layer to adapt the number of input channels.
"""
import torch.nn as nn

class ChannelAdapter(nn.Module):
    """
    A simple yet effective module to project an N-channel input to a C-out channel
    output, typically used to adapt a multi-channel feature bank (N channels) to
    the 3 channels expected by a pretrained encoder.
    """
    def __init__(self, c_in: int, c_out: int = 3, init: str = "xavier"):
        """
        Args:
            c_in: Number of input channels (from the feature bank).
            c_out: Number of output channels (e.g., 3 for ImageNet encoders).
            init: Weight initialization method.
        """
        super().__init__()
        # DEV: Это просто 1x1 свертка. Она учится, как оптимально "смешать"
        # наши каналы с фильтрами в 3-х канальное представление.
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
        
        if init == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            # Default PyTorch initialization is Kaiming Uniform, which is also good.
            pass

    def forward(self, x):
        return self.proj(x)