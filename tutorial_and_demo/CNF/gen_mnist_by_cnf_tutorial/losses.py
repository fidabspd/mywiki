import torch
from typing import List, Tuple, Union


def calculate_gan_discriminator_loss(
    disc_true_output: List[torch.Tensor],
    disc_pred_output: List[torch.Tensor],
) -> Tuple[torch.Tensor]:
    disc_true_output = torch.cat([output.flatten(start_dim=1) for output in disc_true_output], dim=1)
    disc_pred_output = torch.cat([output.flatten(start_dim=1) for output in disc_pred_output], dim=1)
    disc_real_true_loss = torch.log(disc_true_output).sum(dim=1).mean()
    disc_pred_true_loss = torch.log(1 - disc_pred_output).sum(dim=1).mean()
    gan_discriminator_loss = -(disc_real_true_loss + disc_pred_true_loss)
    return gan_discriminator_loss, disc_real_true_loss, disc_pred_true_loss


def calculate_gan_generator_loss(disc_pred_output: torch.Tensor) -> Tuple[torch.Tensor]:
    disc_pred_output = torch.cat([output.flatten(start_dim=1) for output in disc_pred_output], dim=1)
    disc_fake_pred_loss = torch.log(disc_pred_output).sum(dim=1).mean()
    gan_generator_loss = -disc_fake_pred_loss
    return gan_generator_loss, disc_fake_pred_loss


def calculate_disc_fake_feature_map_loss(
    disc_true_feature_maps: List[torch.Tensor], disc_pred_feature_maps: List[torch.Tensor]
) -> torch.Tensor:
    disc_fake_feature_map_loss = 0
    for disc_true_feature_map, disc_pred_feature_map in zip(disc_true_feature_maps, disc_pred_feature_maps):
        disc_fake_feature_map_loss += torch.abs(disc_true_feature_map - disc_pred_feature_map).mean()
    return disc_fake_feature_map_loss


def calculate_vae_loss(
    image_true: torch.Tensor,
    imgae_pred: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_log_probs: torch.Tensor,
) -> Tuple[torch.Tensor]:
    recon_loss = image_true * torch.log(imgae_pred) + (1 - image_true) * torch.log(1 - imgae_pred)
    recon_loss = torch.flatten(recon_loss, start_dim=1).sum(dim=1).mean()
    posterior_log_probs = (
        -torch.log(posterior_std) - 0.5 * torch.log(2 * torch.FloatTensor([torch.pi]).to(posterior_std.device)) - 0.5
    )
    posterior_log_probs = posterior_log_probs.sum(dim=1)  # assume non diagonal elements of cov matrix are zero
    kl_divergence = torch.abs(posterior_log_probs - prior_log_probs)
    kl_divergence = kl_divergence.mean()
    elbo = recon_loss - kl_divergence
    loss = -elbo
    return loss, recon_loss, kl_divergence


def calculate_cnf_loss(logp_x: torch.Tensor) -> Tuple[torch.Tensor]:
    log_prob = logp_x.mean()
    loss = -log_prob
    return loss, log_prob


def calculate_final_discriminator_loss(
    disc_true_output: List[torch.Tensor],
    disc_pred_output: List[torch.Tensor],
    return_only_final_loss: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    final_discriminator_loss, disc_real_true_loss, disc_real_pred_loss = calculate_gan_discriminator_loss(
        disc_true_output, disc_pred_output
    )
    if return_only_final_loss:
        return final_discriminator_loss
    else:
        return final_discriminator_loss, disc_real_true_loss, disc_real_pred_loss


def calculate_final_generator_loss(
    image_true: torch.Tensor,
    imgae_pred: torch.Tensor,
    disc_pred_output: torch.Tensor,
    disc_true_feature_maps: List[torch.Tensor],
    disc_pred_feature_maps: List[torch.Tensor],
    std: torch.Tensor,
    logp_x: torch.Tensor,
    disc_fake_feature_map_loss_weigth: float = 1.0,
    gan_generator_loss_weight: float = 1.0,
    vae_loss_weight: float = 1.0,
    cnf_loss_weight: float = 1.0,
    return_only_final_loss: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    disc_fake_feature_map_loss = calculate_disc_fake_feature_map_loss(disc_true_feature_maps, disc_pred_feature_maps)
    gan_generator_loss, disc_fake_pred_loss = calculate_gan_generator_loss(disc_pred_output)
    vae_loss, recon_loss, kl_divergence = calculate_vae_loss(image_true, imgae_pred, std, logp_x)
    cnf_loss, cnf_log_prob = calculate_cnf_loss(logp_x)
    final_generator_loss = (
        disc_fake_feature_map_loss_weigth * disc_fake_feature_map_loss
        + gan_generator_loss_weight * gan_generator_loss
        + vae_loss_weight * vae_loss
        + cnf_loss_weight * cnf_loss
    )
    if return_only_final_loss:
        return final_generator_loss
    else:
        return (
            final_generator_loss,
            disc_fake_feature_map_loss,
            disc_fake_pred_loss,
            recon_loss,
            kl_divergence,
            cnf_log_prob,
        )
