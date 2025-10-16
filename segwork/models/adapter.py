# segwork/models/adapter.py

"""
A learnable 1x1 convolution layer to adapt the number of input channels.
"""
import torch.nn as nn
import torch

class ChannelAdapter(nn.Module):
    """
    A simple yet effective module to project an N-channel input to a C-out channel
    output, typically used to adapt a multi-channel feature bank (N channels) to
    the 3 channels expected by a pretrained encoder.
    """
    def __init__(
        self,
        c_in: int,
        c_out: int = 3,
        init: str = "xavier",
        channel_names: list[str] | None = None,
        initial_channel_weights: dict | None = None
    ):
        """
        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            init: Weight initialization method ('xavier').
            channel_names: List of input channel names, e.g., ['raw', 'clahe'].
            initial_channel_weights: Optional dict to apply multipliers to initial weights.
        """
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

        # Step 1: Standard weight initialization
        if init == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        
        # Step 2: Apply soft initial weights if configured
        if initial_channel_weights and channel_names:
            if c_in != len(channel_names):
                raise ValueError(f"Channel name count ({len(channel_names)}) does not match adapter input channels ({c_in}).")

            print("Applying soft initial channel weights to adapter...")
            with torch.no_grad():
                default_factor = initial_channel_weights.get('default', 1.0)
                
                for i, channel_name in enumerate(channel_names):
                    factor = initial_channel_weights.get(channel_name, default_factor)
                    if factor != 1.0:
                        # Multiply the weights corresponding to this input channel
                        self.proj.weight[:, i, :, :] *= factor
                        print(f"  - Channel '{channel_name}' weights scaled by {factor}")


    def forward(self, x):
        return self.proj(x)