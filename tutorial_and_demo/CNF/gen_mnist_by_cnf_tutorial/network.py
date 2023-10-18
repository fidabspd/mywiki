import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from typing import Tuple, List


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        condition_dim: int = 4,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        n_hidden_layers: int = 2,
        epsilon: float = 1e-6,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.linear_out = nn.Linear(hidden_dim, latent_dim * 2)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        output = torch.flatten(input, start_dim=1)
        output = self.linear_in(output)
        for i, layer in enumerate(self.linear_hidden):
            output = self.dropout(torch.tanh(layer(output)))
        output = self.linear_out(output)
        mean, std = torch.split(output, self.latent_dim, dim=1)
        std = torch.exp(std) + self.epsilon
        output = mean + torch.randn_like(mean) * std
        return output, mean, std


class ImageDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        condition_dim: int = 4,
        hidden_dim: int = 32,
        out_dim: int = 784,
        n_hidden_layers: int = 2,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(latent_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.linear_in(input)
        for i, layer in enumerate(self.linear_hidden):
            output = self.dropout(torch.relu(layer(output)))
        output = torch.sigmoid(self.linear_out(output))
        output = output.view(-1, 1, int(self.out_dim**0.5), int(self.out_dim**0.5))
        return output


class HyperNetwork(nn.Module):
    """https://arxiv.org/abs/1609.09106"""

    def __init__(self, in_out_dim, hidden_dim, width, condition_dim):
        super().__init__()

        blocksize = width * in_out_dim

        self.linear_hidden_0 = nn.Linear(1, hidden_dim)
        self.linear_hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.linear_hidden_0(params))
        params = torch.tanh(self.linear_hidden_1(params))
        params = self.linear_out(params)

        # restructure
        W = params[:, : self.blocksize].reshape(-1, self.width, self.in_out_dim, 1).transpose(0, 1).contiguous()

        U = (
            params[:, self.blocksize : 2 * self.blocksize]
            .reshape(-1, self.width, 1, self.in_out_dim)
            .transpose(0, 1)
            .contiguous()
        )

        G = (
            params[:, 2 * self.blocksize : 3 * self.blocksize]
            .reshape(-1, self.width, 1, self.in_out_dim)
            .transpose(0, 1)
            .contiguous()
        )
        U = U * torch.sigmoid(G)

        B = params[:, 3 * self.blocksize :].reshape(-1, self.width, 1, 1).transpose(0, 1).contiguous()
        return [W, B, U]


class TanhSigmoidMultiplyCondition(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        in_act = z + condition
        t_act = torch.tanh(in_act[:, : self.hidden_dim])
        s_act = torch.sigmoid(in_act[:, self.hidden_dim :])
        output = t_act * s_act
        return output


class ODEFunc(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, condition_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width, condition_dim)

    def trace_df_dz(self, f, z):
        """Calculates the trace (equals to det) of the Jacobian df/dz."""
        sum_diag = 0.0
        for i in range(z.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

        return sum_diag.contiguous()  # [batch_size]

    def forward(self, t, states):
        z, logp_z = states
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = z.unsqueeze(0).unsqueeze(-2).repeat(self.width, 1, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)  # mean by width dim
            dz_dt = dz_dt.squeeze(dim=1)

            dlogp_z_dt = -self.trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


class PartDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        conv0 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding="same")
        conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding="same")
        conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding="same")
        self.conv_layers = nn.ModuleList([conv0, conv1, conv2])
        self.post_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride)

    def forward(self, input) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feature_maps = []
        for layer in self.conv_layers:
            input = layer(input)
            feature_maps.append(input)
        output = torch.sigmoid(self.post_layer(input))
        return output, feature_maps


class FullDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        discriminators = []
        for kernel_size in (3, 5):
            discriminator = PartDiscriminator(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
            discriminators.append(discriminator)
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, input: torch.Tensor) -> Tuple[List[torch.Tensor]]:
        outputs = []
        feature_maps = []
        for discriminator in self.discriminators:
            output, feature_map = discriminator(input)
            outputs.append(output)
            feature_maps.extend(feature_map)
        return outputs, feature_maps


class VAECNF(nn.Module):
    def __init__(
        self,
        batch_size: int = 64,
        in_out_dim: int = 784,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        condition_dim: int = 4,
        n_hidden_layers: int = 2,
        ode_t0: int = 0,
        ode_t1: int = 10,
        cov_value: float = 0.1,
        ode_hidden_dim: int = 64,
        ode_width: int = 64,
        dropout_ratio: float = 0.1,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.t0 = ode_t0
        self.t1 = ode_t1
        self.latent_dim = latent_dim

        # pdf of z0
        mean = torch.zeros(latent_dim).type(torch.float32)
        cov = torch.zeros(latent_dim, latent_dim).type(torch.float32)
        cov.fill_diagonal_(cov_value)
        self.p_z0 = torch.distributions.MultivariateNormal(
            loc=mean.to(self.device), covariance_matrix=cov.to(self.device)
        )
        self.condition_embedding_layer = nn.Embedding(10, condition_dim)

        self.image_encoder = ImageEncoder(
            in_dim=in_out_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_ratio=dropout_ratio,
        )
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            out_dim=in_out_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_ratio=dropout_ratio,
        )
        self.ode_func = ODEFunc(
            in_out_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            condition_dim=condition_dim,
            width=ode_width,
        )

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        condition = self.condition_embedding_layer(condition)

        z_t1, mean, std = self.image_encoder(input, condition)
        reconstructed = self.image_decoder(z_t1, condition)

        logp_diff_t1 = torch.zeros(self.batch_size, 1).type(torch.float32).to(self.device)
        z_t, logp_diff_t = odeint(
            self.ode_func,
            (z_t1, logp_diff_t1),
            torch.tensor([self.t1, self.t0]).type(torch.float32).to(self.device),  # focus on [T1, T0] (not [T0, T1])
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = self.p_z0.log_prob(z_t0).to(self.device) - logp_diff_t0.view(-1)

        return reconstructed, logp_x, mean, std

    def generate(self, condition: torch.Tensor, n_time_steps: int = 2) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            z_t0 = self.p_z0.sample([1]).to(self.device)
            logp_diff_t0 = torch.zeros(1, 1).type(torch.float32).to(self.device)

            time_space = np.linspace(self.t0, self.t1, n_time_steps)  # [T0, T1] for generation
            z_t_samples, _ = odeint(
                self.ode_func,
                (z_t0, logp_diff_t0),
                torch.tensor(time_space).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method="dopri5",
            )
            z_t_samples = z_t_samples.view(n_time_steps, -1)
            gen_image = self.image_decoder(z_t_samples, condition)

        return gen_image, time_space
