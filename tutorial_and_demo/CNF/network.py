import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


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


class HyperNetwork(nn.Module):
    """https://arxiv.org/abs/1609.09106"""

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[: self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize : 2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize : 3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize :].reshape(self.width, 1, 1)
        return [W, B, U]
    

def trace_df_dz(f, z):
    """Calculates the trace (equals to det) of the Jacobian df/dz."""
    sum_diag = 0.0
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()  # [batch_size]


class ODEFunc(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z, logp_z = states
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)  # mean by width dim

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


class AECNF(nn.Module):
    def __init__(
        self,
        batch_size: int = 64,
        image_size: int = 28,
        in_out_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 16,
        kernel_size: int = 3,
        ode_t0: int = 0,
        ode_t1: int = 10,
        ode_hidden_dim: int = 64,
        ode_width: int = 64,
        dropout_ratio: float = 0.1,
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.t0 = ode_t0
        self.t1 = ode_t1

        # pdf of z0
        mean = torch.zeros(latent_channels).type(torch.float32)
        cov = torch.zeros(latent_channels, latent_channels).type(torch.float32)
        cov.fill_diagonal_(0.1)
        self.p_z0 = torch.distributions.MultivariateNormal(loc=mean.to(self.device), covariance_matrix=cov.to(self.device))

        self.image_encoder = ImageEncoder(
            image_size=image_size,
            in_channels=in_out_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            kernel_size=kernel_size,
            dropout_ratio=dropout_ratio,
        )
        self.image_decoder = ImageDecoder(
            image_size=image_size,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            out_channels=in_out_channels,
            kernel_size=kernel_size,
            dropout_ratio=dropout_ratio,
        )
        self.ode_func = ODEFunc(
            in_out_dim=latent_channels,
            hidden_dim=ode_hidden_dim,
            width=ode_width,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z_t1 = self.image_encoder(input)
        reconstructed = self.image_decoder(z_t1)

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
        x_probs = logp_x.mean(0)

        return reconstructed, x_probs
