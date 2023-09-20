import os
import argparse
import logging
import sys
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from network import AECNF, Discriminator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=20)
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

    parser.add_argument("--gan_feature_map_loss_weigth", type=float, default=1.0)
    parser.add_argument("--gan_generator_loss_weight", type=float, default=1.0)
    parser.add_argument("--vae_loss_weight", type=float, default=1.0)
    parser.add_argument("--cnf_loss_weight", type=float, default=1.0)

    parser.add_argument("--viz", type=bool, default=True)
    parser.add_argument("--n_viz_time_steps", type=int, default=11)
    parser.add_argument("--viz_save_dirpath", type=str, default="./cnf_mnist_viz_result/")
    parser.add_argument("--log_dirpath", type=str, default="./logs/")

    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def get_logger(log_dirpath: str, log_filename: str = "training_log.log"):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging
    logger = logging.getLogger(os.path.basename(log_dirpath))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    if not os.path.exists(log_dirpath):
        os.makedirs(log_dirpath)
    file_handler = logging.FileHandler(os.path.join(log_dirpath, log_filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_gan_discriminator_loss(
    disc_true_output: torch.Tensor,
    disc_pred_output: torch.Tensor,
) -> torch.Tensor:
    disc_real_true_loss = torch.log(disc_true_output)
    disc_real_true_loss = disc_real_true_loss.flatten(start_dim=1).sum(dim=1).mean()
    disc_pred_true_loss = torch.log(1 - disc_pred_output)
    disc_pred_true_loss = disc_pred_true_loss.flatten(start_dim=1).sum(dim=1).mean()
    gan_discriminator_loss = -(disc_real_true_loss + disc_pred_true_loss)
    return gan_discriminator_loss, disc_real_true_loss, disc_pred_true_loss


def calculate_gan_generator_loss(disc_pred_output: torch.Tensor) -> torch.Tensor:
    disc_fake_pred_loss = torch.log(disc_pred_output)
    disc_fake_pred_loss = disc_fake_pred_loss.flatten(start_dim=1).sum(dim=1).mean()
    gan_generator_loss = -disc_fake_pred_loss
    return gan_generator_loss, disc_fake_pred_loss


def calculate_gan_feature_map_loss(
    disc_true_feature_maps: torch.Tensor, disc_pred_feature_maps: torch.Tensor
) -> torch.Tensor:
    gan_feature_map_loss = 0
    for disc_true_feature_map, disc_pred_feature_map in zip(disc_true_feature_maps, disc_pred_feature_maps):
        gan_feature_map_loss += torch.abs(disc_true_feature_map - disc_pred_feature_map).mean()
    return gan_feature_map_loss


def calculate_vae_loss(
    image_true: torch.Tensor,
    imgae_pred: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    recon_loss = image_true * torch.log(imgae_pred) + (1 - image_true) * torch.log(1 - imgae_pred)
    recon_loss = torch.flatten(recon_loss, start_dim=1).sum(dim=1).mean()
    kl_divergence = torch.square(mean) + torch.square(std) - torch.log(torch.square(std)) - 1
    kl_divergence = 0.5 * torch.flatten(kl_divergence, start_dim=1).sum(dim=1).mean()
    elbo = recon_loss - kl_divergence
    loss = -elbo
    return loss, recon_loss, kl_divergence


def calculate_cnf_loss(x_probs: torch.Tensor) -> torch.Tensor:
    loss = -x_probs
    return loss, x_probs


def calculate_final_discriminator_loss(
    disc_true_output: torch.Tensor,
    disc_pred_output: torch.Tensor,
    return_only_final_loss: bool = True,
) -> torch.Tensor:
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
    disc_true_feature_maps: torch.Tensor,
    disc_pred_feature_maps: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    cnf_probs: torch.Tensor,
    gan_feature_map_loss_weigth: float = 1.0,
    gan_generator_loss_weight: float = 1.0,
    vae_loss_weight: float = 1.0,
    cnf_loss_weight: float = 1.0,
    return_only_final_loss: bool = True,
):
    gan_feature_map_loss = calculate_gan_feature_map_loss(disc_true_feature_maps, disc_pred_feature_maps)
    gan_generator_loss, disc_fake_pred_loss = calculate_gan_generator_loss(disc_pred_output)
    vae_loss, recon_loss, kl_divergence = calculate_vae_loss(image_true, imgae_pred, mean, std)
    cnf_loss, cnf_probs = calculate_cnf_loss(cnf_probs)
    final_generator_loss = (
        gan_feature_map_loss_weigth * gan_feature_map_loss
        + gan_generator_loss_weight * gan_generator_loss
        + vae_loss_weight * vae_loss
        + cnf_loss_weight * cnf_loss
    )
    if return_only_final_loss:
        return final_generator_loss
    else:
        return final_generator_loss, gan_feature_map_loss, disc_fake_pred_loss, recon_loss, kl_divergence, cnf_probs


def visualize_inference_result(
    z_t_samples: torch.Tensor,
    condition: int,
    time_space: np.array,
    save_dirpath: str,
    global_step: int,
):
    if not os.path.exists(save_dirpath):
        os.makedirs(save_dirpath)

    fig, ax = plt.subplots(1, 11, figsize=(22, 1.8))
    plt.suptitle(f"condition: {condition}", fontsize=17, y=1.15)
    for i in range(11):
        t = time_space[i]
        z_sample = z_t_samples[i].view(28, 28)
        ax[i].imshow(z_sample.detach().cpu())
        ax[i].set_axis_off()
        ax[i].set_title("$p(\mathbf{z}_{" + str(t) + "})$")
    save_filename = f"infer_{global_step}.png"
    plt.savefig(os.path.join(save_dirpath, save_filename), dpi=300, bbox_inches="tight")


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
    gan_feature_map_loss_weigth: float = 1.0,
    gan_generator_loss_weight: float = 1.0,
    vae_loss_weight: float = 1.0,
    cnf_loss_weight: float = 1.0,
    viz: bool = True,
    n_viz_time_steps: int = 11,
    viz_save_dirpath: str = "./cnf_mnist_viz_result/",
    logger: logging = None
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
            final_discriminator_loss, disc_real_true_loss, disc_real_pred_loss = calculate_final_discriminator_loss(
                disc_true_output, disc_pred_output, return_only_final_loss=False
            )
            final_discriminator_loss.backward()
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()

            # forward discriminator
            disc_true_output, disc_true_feature_maps = discriminator(image)
            disc_pred_output, disc_pred_feature_maps = discriminator(reconstructed)

            # backward generator
            final_generator_loss, gan_feature_map_loss, disc_fake_pred_loss, recon_loss, kl_divergence, cnf_probs = calculate_final_generator_loss(
                image_true=image,
                imgae_pred=reconstructed,
                disc_pred_output=disc_pred_output,
                disc_true_feature_maps=disc_true_feature_maps,
                disc_pred_feature_maps=disc_pred_feature_maps,
                mean=mean,
                std=std,
                cnf_probs=cnf_probs,
                gan_feature_map_loss_weigth=gan_feature_map_loss_weigth,
                gan_generator_loss_weight=gan_generator_loss_weight,
                vae_loss_weight=vae_loss_weight,
                cnf_loss_weight=cnf_loss_weight,
                return_only_final_loss=False,
            )
            final_generator_loss.backward()
            optimizer_generator.step()
            optimizer_generator.zero_grad()

            # result of step
            step_pbar.set_description(
                f"[Global step: {global_step}, Discriminator loss: {final_discriminator_loss:.2f}, Generator loss: {final_generator_loss:.2f}]"
            )

            if global_step % log_interval == 0:
                _info = ""
                _info += f"\n=== Global step: {global_step} ==="
                _info += f"\nTraining Loss"
                _info += f"\n\tfinal_discriminator_loss: {final_discriminator_loss:.2f}"
                _info += f"\n\t\tdisc_real_true_loss: {disc_real_true_loss:.2f}, disc_real_pred_loss: {disc_real_pred_loss:.2f}"
                _info += f"\n\tfinal_generator_loss: {final_generator_loss:.2f}"
                _info += f"\n\t\tgan_feature_map_loss: {gan_feature_map_loss:.2f}, disc_fake_pred_loss: {disc_fake_pred_loss:.2f}"
                _info += f", recon_loss: {recon_loss:.2f}, kl_divergence: {kl_divergence:.2f}, cnf_probs: {cnf_probs:.2f}\n"
                if logger is not None:
                    logger.info(_info)

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

            eval_final_discriminator_loss, eval_disc_real_true_loss, eval_disc_real_pred_loss = calculate_final_discriminator_loss(
                disc_true_output, disc_pred_output, return_only_final_loss=False
            )
            eval_final_generator_loss, eval_gan_feature_map_loss, eval_disc_fake_pred_loss, eval_recon_loss, eval_kl_divergence, eval_cnf_probs = calculate_final_generator_loss(
                image_true=image,
                imgae_pred=reconstructed,
                disc_pred_output=disc_pred_output,
                disc_true_feature_maps=disc_true_feature_maps,
                disc_pred_feature_maps=disc_pred_feature_maps,
                mean=mean,
                std=std,
                cnf_probs=cnf_probs,
                gan_feature_map_loss_weigth=gan_feature_map_loss_weigth,
                gan_generator_loss_weight=gan_generator_loss_weight,
                vae_loss_weight=vae_loss_weight,
                cnf_loss_weight=cnf_loss_weight,
                return_only_final_loss=False,
            )
            break

        _info = ""
        _info += f"\n=== Global step: {global_step} ==="
        _info += f"\n=== Evaluation Loss ==="
        _info += f"\n\tfinal_discriminator_loss: {eval_final_discriminator_loss:.2f}"
        _info += f"\n\t\tdisc_real_true_loss: {eval_disc_real_true_loss:.2f}, disc_real_pred_loss: {eval_disc_real_pred_loss:.2f}"
        _info += f"\n\tfinal_generator_loss: {eval_final_generator_loss:.2f}"
        _info += f"\n\t\tgan_feature_map_loss: {eval_gan_feature_map_loss:.2f}, disc_fake_pred_loss: {eval_disc_fake_pred_loss:.2f}"
        _info += f", recon_loss: {eval_recon_loss:.2f}, kl_divergence: {eval_kl_divergence:.2f}, cnf_probs: {eval_cnf_probs:.2f}\n"
        if logger is not None:
            logger.info(_info)

        if viz:
            condition = label[:1]
            z_t_samples, time_space = generator.generate(condition, n_viz_time_steps)
            condition = condition[0].cpu().item()
            visualize_inference_result(z_t_samples, condition, time_space, viz_save_dirpath, global_step)

    global_step = -1

    epoch_pbar = tqdm(range(n_epochs))
    for epoch_idx in epoch_pbar:
        epoch_pbar.set_description(f"Epoch: {epoch_idx}")
        train_and_evaluate_one_epoch()


def main(args):
    logger = get_logger(args.log_dirpath)
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
    discriminator = Discriminator(
        in_channels=1,
        hidden_channels=32,
        kernel_size=3,
        stride=1,
    ).to(args.device)
    generator_n_params = count_parameters(generator)
    discriminator_n_params = count_parameters(discriminator)
    print(f"generator_n_params: {generator_n_params}, discriminator_n_params: {discriminator_n_params}")
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
        gan_feature_map_loss_weigth=args.gan_feature_map_loss_weigth,
        gan_generator_loss_weight=args.gan_generator_loss_weight,
        vae_loss_weight=args.vae_loss_weight,
        cnf_loss_weight=args.cnf_loss_weight,
        viz=args.viz,
        n_viz_time_steps=args.n_viz_time_steps,
        viz_save_dirpath=args.viz_save_dirpath,
        logger=logger,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
