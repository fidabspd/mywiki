import os
import sys
import logging
import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_logger(log_dirpath: str, log_filename: str = "training_log.log") -> logging:
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


def dict_to_indented_str(input: dict, indent_space: int = 4) -> str:
    def recur_func(input, n_indent, indent_space):
        nonlocal output
        output += " " * indent_space * n_indent + "{\n"
        for k, v in input.items():
            if isinstance(v, dict):
                recur_func(v, n_indent=n_indent + 1, indent_space=indent_space)
            else:
                k = f'"{k}"'
                if isinstance(v, str):
                    v = f'"{v}"'
                elif isinstance(v, bool):
                    v = str(v).lower()
                output += " " * indent_space * (n_indent + 1) + f"{k}: {v},\n"
        output += " " * indent_space * n_indent + "}\n"

    output = ""
    recur_func(input, 0, indent_space=indent_space)
    return output


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_and_get_grad_values(
    model: torch.nn.Module, max_clip_value: float = None, norm_type: float = 2.0
) -> torch.Tensor:
    parameters = model.parameters()
    if not max_clip_value:
        max_clip_value = torch.inf
    total_norm = torch.nn.utils.clip_grad_norm_(parameters=parameters, max_norm=max_clip_value, norm_type=norm_type)
    return total_norm


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim,
    epoch: int,
    global_step: int,
    checkpoint_filepath: str,
    logger: logging = None,
) -> None:
    if logger is not None:
        logger.info(
            "Saving model and optimizer state at global step {} to {}".format(global_step, checkpoint_filepath)
        )

    checkpoint_dirpath = os.path.split(checkpoint_filepath)[0]
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    optimizer_state_dict = None
    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()

    state_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": state_dict,
            "optimizer": optimizer_state_dict,
        },
        checkpoint_filepath,
    )


def load_checkpoint(
    checkpoint_filepath: str, model: torch.nn.Module, optimizer: torch.optim, logger: logging = None
) -> tuple:
    saved_checkpoint = torch.load(checkpoint_filepath)

    epoch = saved_checkpoint["epoch"]
    global_step = saved_checkpoint["global_step"]
    model.load_state_dict(saved_checkpoint["model"])
    optimizer.load_state_dict(saved_checkpoint["optimizer"])

    if logger is not None:
        logger.info(f"Load checkpoint {checkpoint_filepath}")

    return epoch, global_step, model, optimizer


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
    for i in range(11):
        t = time_space[i]
        z_sample = z_t_samples[i].view(28, 28)
        ax[i].imshow(z_sample.detach().cpu())
        ax[i].set_axis_off()
        ax[i].set_title("$p(\mathbf{z}_{" + str(t) + "})$")
    save_filename = f"infer_{global_step}.png"
    plt.savefig(os.path.join(save_dirpath, save_filename), dpi=300, bbox_inches="tight")
    plt.close()
