import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=32,
        latent_channels=16,
        conv_kernel_size=3,
        conv_stride=1,
        pool_kernel_size=3,
        pool_stride=2,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, conv_kernel_size, conv_stride)
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, conv_kernel_size, conv_stride)
        self.conv2 = nn.Conv2d(hidden_channels, latent_channels, conv_kernel_size, conv_stride)
        self.pool = nn.AvgPool2d(pool_kernel_size, pool_stride, padding=0)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input):
        batch_size = input.size(0)
        output = self.dropout(torch.tanh(self.conv0(input)))
        output = self.pool(output)
        output = self.dropout(torch.tanh(self.conv1(output)))
        output = self.pool(output)
        output = self.conv2(output)
        output = output.view(batch_size, -1)
        return output
