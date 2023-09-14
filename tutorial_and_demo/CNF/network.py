import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 16,
        kernel_size: int = 3,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, kernel_size, 1)
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear((image_size - (kernel_size - 1) * 3) ** 2 * hidden_channels, latent_channels)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.dropout(torch.tanh(self.conv0(input)))
        output = self.dropout(torch.tanh(self.conv1(output)))
        output = self.dropout(torch.tanh(self.conv2(output)))
        output = self.flatten(output)
        output = self.linear(output)
        return output


class ImageDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        latent_channels: int = 16,
        hidden_channels: int = 32,
        out_channels: int = 1,
        kernel_size: int = 3,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_image_size = image_size - 3 * (kernel_size - 1)
        self.linear = nn.Linear(latent_channels, (image_size - (kernel_size - 1) * 3) ** 2 * hidden_channels)
        self.convt0 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, 1)
        self.convt1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, 1)
        self.convt2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size, 1)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.dropout(torch.tanh(self.linear(input)))
        output = output.view(-1, self.hidden_channels, self.latent_image_size, self.latent_image_size)
        output = self.dropout(torch.tanh(self.convt0(output)))
        output = self.dropout(torch.tanh(self.convt1(output)))
        output = self.convt2(output)
        return output


class AECNF(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 16,
        kernel_size: int = 3,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            kernel_size=kernel_size,
            dropout_ratio=dropout_ratio,
        )
        self.image_decoder = ImageDecoder(
            image_size=image_size,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        latent = self.image_encoder(input)
        output = self.image_decoder(latent)
        return output
