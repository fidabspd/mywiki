import torch
from torch import nn
from typing import List, Tuple, Union


class FinalDiscriminatorLoss(nn.Module):
    def __init__(self, return_only_final_loss: bool = True) -> None:
        super().__init__()
        self.return_only_final_loss = return_only_final_loss

    def calculate_gan_discriminator_loss(
        self,
        disc_true_output: List[torch.Tensor],
        disc_pred_output: List[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        disc_true_output = torch.cat([output.flatten(start_dim=1) for output in disc_true_output], dim=1)
        disc_pred_output = torch.cat([output.flatten(start_dim=1) for output in disc_pred_output], dim=1)
        disc_real_true_loss = torch.log(disc_true_output).sum(dim=1).mean()
        disc_real_pred_loss = torch.log(1 - disc_pred_output).sum(dim=1).mean()
        gan_discriminator_loss = -(disc_real_true_loss + disc_real_pred_loss)
        return gan_discriminator_loss, disc_real_true_loss, disc_real_pred_loss

    def forward(
        self,
        disc_true_output: List[torch.Tensor],
        disc_pred_output: List[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        final_discriminator_loss, disc_real_true_loss, disc_real_pred_loss = self.calculate_gan_discriminator_loss(
            disc_true_output, disc_pred_output
        )
        if self.return_only_final_loss:
            return final_discriminator_loss
        else:
            return final_discriminator_loss, disc_real_true_loss, disc_real_pred_loss


class FinalGeneratorLoss(nn.Module):
    def __init__(
        self,
        disc_fake_feature_map_loss_weigth: float = 1.0,
        gan_generator_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        kl_divergence_weight: float = 1.0,
        cnf_loss_weight: float = 1.0,
        return_only_final_loss: bool = True,
    ) -> None:
        super().__init__()
        self.disc_fake_feature_map_loss_weigth = disc_fake_feature_map_loss_weigth
        self.gan_generator_loss_weight = gan_generator_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.kl_divergence_weight = kl_divergence_weight
        self.cnf_loss_weight = cnf_loss_weight
        self.return_only_final_loss = return_only_final_loss

    def calculate_gan_generator_loss(self, disc_pred_output: torch.Tensor) -> Tuple[torch.Tensor]:
        disc_pred_output = torch.cat([output.flatten(start_dim=1) for output in disc_pred_output], dim=1)
        disc_fake_pred_loss = torch.log(disc_pred_output).sum(dim=1).mean()
        gan_generator_loss = -disc_fake_pred_loss
        return gan_generator_loss, disc_fake_pred_loss

    def calculate_disc_fake_feature_map_loss(
        self, disc_true_feature_maps: List[torch.Tensor], disc_pred_feature_maps: List[torch.Tensor]
    ) -> torch.Tensor:
        disc_fake_feature_map_loss = 0
        for disc_true_feature_map, disc_pred_feature_map in zip(disc_true_feature_maps, disc_pred_feature_maps):
            disc_fake_feature_map_loss += torch.abs(disc_true_feature_map - disc_pred_feature_map).mean()
        return disc_fake_feature_map_loss

    def calculate_recon_loss(self, image_true: torch.Tensor, image_pred: torch.Tensor) -> torch.Tensor:
        recon_loss = image_true * torch.log(image_pred + 1e-7) + (1 - image_true) * torch.log(1 - image_pred + 1e-7)
        recon_loss = torch.flatten(recon_loss, start_dim=1).sum(dim=1).mean()
        return recon_loss

    # def calculate_kl_divergence(
    #     self, mean: torch.Tensor, std: torch.Tensor
    # ) -> torch.Tensor:
    #     kl_divergence = 0.5 * (-2 * torch.log(std) + torch.square(std) + torch.square(mean) - 1)
    #     kl_divergence = torch.flatten(kl_divergence, start_dim=1).sum(dim=1).mean()
    #     return kl_divergence

    def calculate_kl_divergence(self, posterior_std: torch.Tensor, prior_log_probs: torch.Tensor) -> torch.Tensor:
        posterior_log_probs = (
            -torch.log(posterior_std)
            - 0.5 * torch.log(2 * torch.FloatTensor([torch.pi]).to(posterior_std.device))
            - 0.5
        )
        posterior_log_probs = posterior_log_probs.sum(dim=1)  # assume non diagonal elements of cov matrix are zero
        kl_divergence = posterior_log_probs - prior_log_probs
        kl_divergence = kl_divergence.mean()
        return kl_divergence

    def calculate_cnf_loss(self, logp_x: torch.Tensor) -> Tuple[torch.Tensor]:
        log_prob = logp_x.mean()
        loss = -log_prob
        return loss, log_prob

    def forward(
        self,
        image_true: torch.Tensor,
        image_pred: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        logp_x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        recon_loss = self.calculate_recon_loss(image_true, image_pred)
        kl_divergence = self.calculate_kl_divergence(std, logp_x)
        cnf_loss, cnf_log_prob = self.calculate_cnf_loss(logp_x)
        final_generator_loss = (
            -self.recon_loss_weight * recon_loss
            + self.kl_divergence_weight * kl_divergence
            # + self.cnf_loss_weight * cnf_loss
        )
        if self.return_only_final_loss:
            return final_generator_loss
        else:
            return (
                final_generator_loss,
                recon_loss,
                kl_divergence,
                cnf_log_prob,
            )
