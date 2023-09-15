import argparse
from tqdm import tqdm
import torch
import torchvision

from network import AECNF


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout_ratio", type=int, default=0.1)

    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_loss(
    image_true: torch.Tensor,
    imgae_pred: torch.Tensor,
    x_probs: torch.Tensor,
):
    recon_loss = ((image_true - imgae_pred) ** 2).sum(dim=(1, 2, 3)).mean()
    cnf_loss = -x_probs
    total_loss = recon_loss + cnf_loss
    return total_loss


def train_and_evaluate(
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_dl: torch.utils.data.DataLoader,
    eval_dl: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    eval_interval: int = 200,
):
    def train_and_evaluate_one_epoch():
        nonlocal global_step

        step_pbar = tqdm(train_dl)
        for batch in step_pbar:
            global_step += 1
            model.train()

            image, label = batch
            image, label = image.to(args.device), label.to(args.device)
            reconstructed, x_probs = model(image)

            total_loss = calculate_loss(image, reconstructed, x_probs)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if global_step % eval_interval == 0:
            #     evaluate()

    def evaluate():
        model.eval()
        raise NotImplementedError

    global_step = -1

    epoch_pbar = tqdm(range(n_epochs))
    for epoch_idx in epoch_pbar:
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

    model = AECNF().to(args.device)
    n_params = count_parameters(model)
    print(f"n_params: {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_and_evaluate(
        model=model,
        optimizer=optimizer,
        train_dl=train_dl,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
