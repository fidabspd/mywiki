import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from network import AECNF


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.002)

    parser.add_argument("--in_out_dim", type=int, default=784)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--ode_t0", type=int, default=0)
    parser.add_argument("--ode_t1", type=int, default=10)
    parser.add_argument("--ode_hidden_dim", type=int, default=32)
    parser.add_argument("--ode_width", type=int, default=32)
    parser.add_argument("--dropout_ratio", type=float, default=0.1)

    parser.add_argument("--cnf_loss_weight", type=float, default=1.5)

    parser.add_argument("--viz", type=bool, default=True)
    parser.add_argument("--n_viz_time_steps", type=int, default=11)
    parser.add_argument("--viz_save_dirpath", type=str, default="./cnf_mnist_viz_result/")

    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_loss(
    image_true: torch.Tensor,
    imgae_pred: torch.Tensor,
    x_probs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    cnf_loss_weight: float = 3.0,
):
    recon_loss = image_true * torch.log(imgae_pred) + (1 - image_true) * torch.log(1 - imgae_pred)
    recon_loss = torch.flatten(recon_loss, start_dim=1).sum(dim=1).mean()
    kl_divergence = torch.square(mean) + torch.square(std) - torch.log(torch.square(std)) - 1
    kl_divergence = 0.5 * torch.flatten(kl_divergence, start_dim=1).sum(dim=1).mean()
    elbo = recon_loss - kl_divergence
    cnf_loss = -x_probs
    vae_loss = -elbo
    total_loss = vae_loss + cnf_loss_weight * cnf_loss
    return total_loss, recon_loss, kl_divergence, cnf_loss


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
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_dl: torch.utils.data.DataLoader,
    eval_dl: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    eval_interval: int = 200,
    cnf_loss_weight: float = 3.0,
    viz: bool = True,
    n_viz_time_steps: int = 11,
    viz_save_dirpath: str = "./cnf_mnist_viz_result/",
):
    def train_and_evaluate_one_epoch():
        nonlocal global_step

        step_pbar = tqdm(train_dl)
        for batch in step_pbar:
            global_step += 1
            model.train()

            image, label = batch
            image, label = image.to(args.device), label.to(args.device)
            reconstructed, x_probs, mean, std = model(image, label)

            total_loss, recon_loss, kl_divergence, cnf_loss = calculate_loss(
                image, reconstructed, x_probs, mean, std, cnf_loss_weight
            )
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_pbar.set_description(f"[Train total loss: {total_loss:.2f}]")

            if global_step % eval_interval == 0:
                eval_total_loss, eval_recon_loss, eval_kl_divergence, eval_cnf_loss = evaluate()
                print(f"\n\n\n[Global step: {global_step}]")
                print(
                    "train loss:",
                    f"total_loss: {total_loss}, recon_loss: {recon_loss}, kl_divergence: {kl_divergence}, cnf_loss: {cnf_loss}",
                    sep="\n\t",
                )
                print(
                    "eval loss:",
                    f"total_loss: {eval_total_loss}, recon_loss: {eval_recon_loss}, kl_divergence: {eval_kl_divergence}, cnf_loss: {eval_cnf_loss}",
                    sep="\n\t",
                )

    def evaluate():
        model.eval()
        # eval_step_pbar = tqdm(eval_dl)
        # for batch in eval_step_pbar:
        for batch in eval_dl:
            image, label = batch
            image, label = image.to(args.device), label.to(args.device)
            reconstructed, x_probs, mean, std = model(image, label)

            total_loss, recon_loss, kl_divergence, cnf_loss = calculate_loss(
                image, reconstructed, x_probs, mean, std, cnf_loss_weight
            )
            break

        if viz:
            condition = label[:1]
            z_t_samples, time_space = model.generate(condition, n_viz_time_steps)
            condition = condition[0].cpu().item()
            visualize_inference_result(z_t_samples, condition, time_space, viz_save_dirpath, global_step)

        return total_loss, recon_loss, kl_divergence, cnf_loss

    global_step = -1

    epoch_pbar = tqdm(range(n_epochs))
    for epoch_idx in epoch_pbar:
        epoch_pbar.set_description(f"Epoch: {epoch_idx}")
        train_and_evaluate_one_epoch()


def main(args):
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(args.data_dirpath, transform=mnist_transform, train=True, download=True)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    model = AECNF(
        batch_size=args.batch_size,
        in_out_dim=args.in_out_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        ode_t0=args.ode_t0,
        ode_t1=args.ode_t1,
        ode_hidden_dim=args.ode_hidden_dim,
        ode_width=args.ode_width,
        dropout_ratio=args.dropout_ratio,
        device=args.device,
    ).to(args.device)
    n_params = count_parameters(model)
    print(f"n_params: {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_and_evaluate(
        model=model,
        optimizer=optimizer,
        train_dl=train_dl,
        eval_dl=train_dl,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
        cnf_loss_weight=args.cnf_loss_weight,
        viz=args.viz,
        n_viz_time_steps=args.n_viz_time_steps,
        viz_save_dirpath=args.viz_save_dirpath,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
