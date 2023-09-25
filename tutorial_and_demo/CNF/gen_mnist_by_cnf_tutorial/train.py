import os
import argparse
import json
import logging
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import utils
from network import AECNF, FullDiscriminator
import losses


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.002)

    parser.add_argument("--in_out_dim", type=int, default=784)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--ode_t0", type=int, default=0)
    parser.add_argument("--ode_t1", type=int, default=10)
    parser.add_argument("--cov_value", type=float, default=0.1)
    parser.add_argument("--ode_hidden_dim", type=int, default=32)
    parser.add_argument("--ode_width", type=int, default=32)
    parser.add_argument("--dropout_ratio", type=float, default=0.1)

    parser.add_argument("--disc_fake_feature_map_loss_weigth", type=float, default=1.0)
    parser.add_argument("--gan_generator_loss_weight", type=float, default=1.0)
    parser.add_argument("--vae_loss_weight", type=float, default=1.0)
    parser.add_argument("--cnf_loss_weight", type=float, default=1.0)

    parser.add_argument("--viz", type=bool, default=True)
    parser.add_argument("--n_viz_time_steps", type=int, default=11)
    parser.add_argument("--log_dirpath", type=str, default="./logs/")

    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def train_and_evaluate(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_generator: torch.optim,
    optimizer_discriminator: torch.optim,
    train_dl: torch.utils.data.DataLoader,
    eval_dl: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    log_interval: int = 20,
    eval_interval: int = 200,
    disc_fake_feature_map_loss_weigth: float = 1.0,
    gan_generator_loss_weight: float = 1.0,
    vae_loss_weight: float = 1.0,
    cnf_loss_weight: float = 1.0,
    viz: bool = True,
    n_viz_time_steps: int = 11,
    viz_save_dirpath: str = "./results/",
    logger: logging = None,
    tensorabord_train_writer: SummaryWriter = None,
    tensorabord_eval_writer: SummaryWriter = None,
):
    def train_and_evaluate_one_epoch():
        nonlocal global_step

        step_pbar = tqdm(train_dl)
        for batch in step_pbar:
            global_step += 1
            generator.train()
            discriminator.train()

            # batch data
            image, label = batch
            image, label = image.to(args.device), label.to(args.device)

            # forward generator
            reconstructed, cnf_probs, mean, std = generator(image, label)

            # forward discriminator
            disc_true_output, disc_true_feature_maps = discriminator(image)
            disc_pred_output, disc_pred_feature_maps = discriminator(reconstructed.detach())

            # backward discriminator
            (
                final_discriminator_loss,
                disc_real_true_loss,
                disc_real_pred_loss,
            ) = losses.calculate_final_discriminator_loss(
                disc_true_output, disc_pred_output, return_only_final_loss=False
            )
            final_discriminator_loss.backward()
            grad_discriminator = utils.clip_and_get_grad_values(discriminator)
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()

            # forward discriminator
            disc_true_output, disc_true_feature_maps = discriminator(image)
            disc_pred_output, disc_pred_feature_maps = discriminator(reconstructed)

            # backward generator
            (
                final_generator_loss,
                disc_fake_feature_map_loss,
                disc_fake_pred_loss,
                recon_loss,
                kl_divergence,
                cnf_probs,
            ) = losses.calculate_final_generator_loss(
                image_true=image,
                imgae_pred=reconstructed,
                disc_pred_output=disc_pred_output,
                disc_true_feature_maps=disc_true_feature_maps,
                disc_pred_feature_maps=disc_pred_feature_maps,
                mean=mean,
                std=std,
                cnf_probs=cnf_probs,
                disc_fake_feature_map_loss_weigth=disc_fake_feature_map_loss_weigth,
                gan_generator_loss_weight=gan_generator_loss_weight,
                vae_loss_weight=vae_loss_weight,
                cnf_loss_weight=cnf_loss_weight,
                return_only_final_loss=False,
            )
            final_generator_loss.backward()
            grad_generator = utils.clip_and_get_grad_values(generator)
            optimizer_generator.step()
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            # result of step
            step_pbar.set_description(
                f"[Global step: {global_step}, Discriminator loss: {final_discriminator_loss:.2f}, Generator loss: {final_generator_loss:.2f}]"
            )

            if global_step % log_interval == 0:
                # text logging
                _info = ""
                _info += f"\n=== Global step: {global_step} ==="
                _info += f"\nGradient"
                _info += f"\n\tgrad_generator: {grad_generator:.2f}, grad_discriminator: {grad_discriminator:.2f}"
                _info += f"\nTraining Loss"
                _info += f"\n\tfinal_discriminator_loss: {final_discriminator_loss:.2f}"
                _info += f"\n\t\tdisc_real_true_loss: {disc_real_true_loss:.2f}, disc_real_pred_loss: {disc_real_pred_loss:.2f}"
                _info += f"\n\tfinal_generator_loss: {final_generator_loss:.2f}"
                _info += f"\n\t\tdisc_fake_feature_map_loss: {disc_fake_feature_map_loss:.2f}, disc_fake_pred_loss: {disc_fake_pred_loss:.2f}"
                _info += (
                    f", recon_loss: {recon_loss:.2f}, kl_divergence: {kl_divergence:.2f}, cnf_probs: {cnf_probs:.2f}\n"
                )
                if logger is not None:
                    logger.info(_info)

                # tensorboard logging
                scalar_dict = {}
                scalar_dict.update({"grad/grad_generator": grad_generator})
                scalar_dict.update({"grad/grad_discriminator": grad_discriminator})
                scalar_dict.update({"final_discriminator_loss": final_discriminator_loss})
                scalar_dict.update({"final_discriminator_loss/disc_real_true_loss": disc_real_true_loss})
                scalar_dict.update({"final_discriminator_loss/disc_real_pred_loss": disc_real_pred_loss})
                scalar_dict.update({"final_generator_loss": final_generator_loss})
                scalar_dict.update({"final_generator_loss/disc_fake_feature_map_loss": disc_fake_feature_map_loss})
                scalar_dict.update({"final_generator_loss/disc_fake_pred_loss": disc_fake_pred_loss})
                scalar_dict.update({"final_generator_loss/recon_loss": recon_loss})
                scalar_dict.update({"final_generator_loss/kl_divergence": kl_divergence})
                scalar_dict.update({"final_generator_loss/cnf_probs": cnf_probs})
                if tensorabord_train_writer is not None:
                    for k, v in scalar_dict.items():
                        tensorabord_train_writer.add_scalar(k, v, global_step)

            if global_step % eval_interval == 0:
                evaluate()

    def evaluate():
        generator.eval()
        for batch in eval_dl:
            image, label = batch
            image, label = image.to(args.device), label.to(args.device)
            reconstructed, cnf_probs, mean, std = generator(image, label)
            disc_true_output, disc_true_feature_maps = discriminator(image)
            disc_pred_output, disc_pred_feature_maps = discriminator(reconstructed.detach())

            (
                eval_final_discriminator_loss,
                eval_disc_real_true_loss,
                eval_disc_real_pred_loss,
            ) = losses.calculate_final_discriminator_loss(
                disc_true_output, disc_pred_output, return_only_final_loss=False
            )
            (
                eval_final_generator_loss,
                eval_disc_fake_feature_map_loss,
                eval_disc_fake_pred_loss,
                eval_recon_loss,
                eval_kl_divergence,
                eval_cnf_probs,
            ) = losses.calculate_final_generator_loss(
                image_true=image,
                imgae_pred=reconstructed,
                disc_pred_output=disc_pred_output,
                disc_true_feature_maps=disc_true_feature_maps,
                disc_pred_feature_maps=disc_pred_feature_maps,
                mean=mean,
                std=std,
                cnf_probs=cnf_probs,
                disc_fake_feature_map_loss_weigth=disc_fake_feature_map_loss_weigth,
                gan_generator_loss_weight=gan_generator_loss_weight,
                vae_loss_weight=vae_loss_weight,
                cnf_loss_weight=cnf_loss_weight,
                return_only_final_loss=False,
            )
            break

        # text logging
        _info = ""
        _info += f"\n=== Global step: {global_step} ==="
        _info += f"\nEvaluation Loss"
        _info += f"\n\tfinal_discriminator_loss: {eval_final_discriminator_loss:.2f}"
        _info += f"\n\t\tdisc_real_true_loss: {eval_disc_real_true_loss:.2f}, disc_real_pred_loss: {eval_disc_real_pred_loss:.2f}"
        _info += f"\n\tfinal_generator_loss: {eval_final_generator_loss:.2f}"
        _info += f"\n\t\tdisc_fake_feature_map_loss: {eval_disc_fake_feature_map_loss:.2f}, disc_fake_pred_loss: {eval_disc_fake_pred_loss:.2f}"
        _info += f", recon_loss: {eval_recon_loss:.2f}, kl_divergence: {eval_kl_divergence:.2f}, cnf_probs: {eval_cnf_probs:.2f}\n"
        if logger is not None:
            logger.info(_info)

        # tensorboard logging
        scalar_dict = {}
        scalar_dict.update({"final_discriminator_loss": eval_final_discriminator_loss})
        scalar_dict.update({"final_discriminator_loss/disc_real_true_loss": eval_disc_real_true_loss})
        scalar_dict.update({"final_discriminator_loss/disc_real_pred_loss": eval_disc_real_pred_loss})
        scalar_dict.update({"final_generator_loss": eval_final_generator_loss})
        scalar_dict.update({"final_generator_loss/disc_fake_feature_map_loss": eval_disc_fake_feature_map_loss})
        scalar_dict.update({"final_generator_loss/disc_fake_pred_loss": eval_disc_fake_pred_loss})
        scalar_dict.update({"final_generator_loss/recon_loss": eval_recon_loss})
        scalar_dict.update({"final_generator_loss/kl_divergence": eval_kl_divergence})
        scalar_dict.update({"final_generator_loss/cnf_probs": eval_cnf_probs})
        if tensorabord_eval_writer is not None:
            for k, v in scalar_dict.items():
                tensorabord_eval_writer.add_scalar(k, v, global_step)

        # visualization
        if viz:
            condition = label[:1]
            z_t_samples, time_space = generator.generate(condition, n_viz_time_steps)
            condition = condition[0].cpu().item()
            utils.visualize_inference_result(z_t_samples, condition, time_space, viz_save_dirpath, global_step)

    global_step = -1

    epoch_pbar = tqdm(range(n_epochs))
    for epoch_idx in epoch_pbar:
        epoch_pbar.set_description(f"Epoch: {epoch_idx}")
        train_and_evaluate_one_epoch()


def main(args):
    logger = utils.get_logger(args.log_dirpath)
    tensorboard_train_writer = SummaryWriter(log_dir=os.path.join(args.log_dirpath, "train"))
    tensorboard_eval_writer = SummaryWriter(log_dir=os.path.join(args.log_dirpath, "eval"))

    args_for_logging = utils.dict_to_indented_str(vars(args))
    logger.info(args_for_logging)

    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(args.data_dirpath, transform=mnist_transform, train=True, download=True)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    generator = AECNF(
        batch_size=args.batch_size,
        in_out_dim=args.in_out_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        ode_t0=args.ode_t0,
        ode_t1=args.ode_t1,
        cov_value=args.cov_value,
        ode_hidden_dim=args.ode_hidden_dim,
        ode_width=args.ode_width,
        dropout_ratio=args.dropout_ratio,
        device=args.device,
    ).to(args.device)
    discriminator = FullDiscriminator(
        in_channels=1,
        hidden_channels=32,
        out_channels=1,
        stride=1,
    ).to(args.device)
    generator_n_params = utils.count_parameters(generator)
    discriminator_n_params = utils.count_parameters(discriminator)
    logger.info(f"generator_n_params: {generator_n_params}, discriminator_n_params: {discriminator_n_params}")
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    train_and_evaluate(
        generator=generator,
        discriminator=discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        train_dl=train_dl,
        eval_dl=train_dl,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        disc_fake_feature_map_loss_weigth=args.disc_fake_feature_map_loss_weigth,
        gan_generator_loss_weight=args.gan_generator_loss_weight,
        vae_loss_weight=args.vae_loss_weight,
        cnf_loss_weight=args.cnf_loss_weight,
        viz=args.viz,
        n_viz_time_steps=args.n_viz_time_steps,
        viz_save_dirpath=os.path.join(args.log_dirpath, "viz"),
        logger=logger,
        tensorabord_train_writer=tensorboard_train_writer,
        tensorabord_eval_writer=tensorboard_eval_writer,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
