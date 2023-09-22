import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    plt.close()
