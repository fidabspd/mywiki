import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        epsilon: float = 1e-6,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.linear0 = nn.Linear(in_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim * 2)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.flatten(input, start_dim=1)
        output = self.dropout(torch.nn.functional.elu(self.linear0(output)))
        output = self.dropout(torch.tanh(self.linear1(output)))
        output = self.linear2(output)
        mean, std = torch.split(output, self.latent_dim, dim=1)
        std = torch.exp(std) + self.epsilon
        output = mean + torch.randn_like(mean) * std
        return output, mean, std


class ImageDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        out_dim: int = 784,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.linear0 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.dropout(torch.nn.functional.elu(self.linear0(input)))
        output = self.dropout(torch.tanh(self.linear1(output)))
        output = torch.sigmoid(self.linear2(output))
        output = output.view(-1, 1, int(self.out_dim**0.5), int(self.out_dim**0.5))
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


class TanhSigmoidMultiplyCondition(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        condition_dim: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.linear_latent = nn.Linear(latent_dim, latent_dim * 2)
        self.linear_condition = nn.Linear(condition_dim, latent_dim * 2)

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        z = self.linear_latent(z)
        condition = self.linear_condition(condition)
        in_act = z + condition
        t_act = torch.tanh(in_act[:, : self.latent_dim])
        s_act = torch.sigmoid(in_act[:, self.latent_dim :])
        output = t_act * s_act
        return output


class ODEFunc(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, condition_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)
        self.condition_layer = TanhSigmoidMultiplyCondition(
            latent_dim=in_out_dim,
            condition_dim=condition_dim,
        )

    def forward(self, t, states):
        z, logp_z, condition = states
        batchsize = z.shape[0]
        z = self.condition_layer(z, condition)

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)  # mean by width dim

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt, condition)


class AECNF(nn.Module):
    def __init__(
        self,
        batch_size: int = 64,
        in_out_dim: int = 784,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        condition_dim: int = 4,
        ode_t0: int = 0,
        ode_t1: int = 10,
        cov_value: float = 1.0,
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
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout_ratio=dropout_ratio,
        )
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=in_out_dim,
            dropout_ratio=dropout_ratio,
        )
        self.ode_func = ODEFunc(
            in_out_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            condition_dim=condition_dim,
            width=ode_width,
        )

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z_t1, mean, std = self.image_encoder(input)
        reconstructed = self.image_decoder(z_t1)

        condition_embedding = self.condition_embedding_layer(condition)

        logp_diff_t1 = torch.zeros(self.batch_size, 1).type(torch.float32).to(self.device)
        z_t, logp_diff_t, condition_embedding = odeint(
            self.ode_func,
            (z_t1, logp_diff_t1, condition_embedding),
            torch.tensor([self.t1, self.t0]).type(torch.float32).to(self.device),  # focus on [T1, T0] (not [T0, T1])
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = self.p_z0.log_prob(z_t0).to(self.device) - logp_diff_t0.view(-1)
        x_probs = logp_x.mean(0)

        return reconstructed, x_probs, mean, std

    def generate(self, condition: torch.Tensor, n_time_steps: int = 2) -> torch.Tensor:
        with torch.no_grad():
            z_t0 = self.p_z0.sample([1]).to(self.device)
            logp_diff_t0 = torch.zeros(1, 1).type(torch.float32).to(self.device)

            condition_embedding = self.condition_embedding_layer(condition)

            time_space = np.linspace(self.t0, self.t1, n_time_steps)  # [T0, T1] for generation
            z_t_samples, _, condition_embedding = odeint(
                self.ode_func,
                (z_t0, logp_diff_t0, condition_embedding),
                torch.tensor(time_space).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method="dopri5",
            )
            gen_image = self.image_decoder(z_t_samples)

        return gen_image, time_space
