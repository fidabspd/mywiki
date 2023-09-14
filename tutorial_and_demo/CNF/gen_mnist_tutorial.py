import argparse
from tqdm import tqdm
import torch
import torchvision


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main(args):
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    ds = torchvision.datasets.MNIST(args.data_dirpath, transform=mnist_transform, train=True, download=True)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    epoch_pbar = tqdm(range(args.n_epochs))
    for epoch_idx in epoch_pbar:
        step_pbar = tqdm(dl)
        for batch in step_pbar:
            pass


if __name__ == "__main__":
    args = get_args()
    main(args)
