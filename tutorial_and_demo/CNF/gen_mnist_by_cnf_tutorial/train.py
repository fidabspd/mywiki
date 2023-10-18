import os
import argparse
import json
import logging
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import utils
from network import VAECNF, FullDiscriminator
import losses


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--data_dirpath", type=str, default="./.data/")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=35)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=100)
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
    parser.add_argument("--recon_loss_weight", type=float, default=3.0)
    parser.add_argument("--kl_divergence_weight", type=float, default=0.1)
    parser.add_argument("--cnf_loss_weight", type=float, default=0.1)

    parser.add_argument("--viz", type=bool, default=True)
    parser.add_argument("--n_viz_time_steps", type=int, default=11)
    parser.add_argument("--log_dirpath", type=str, default="./logs/vae_cnf_notcond_ver2_final/")

    parser.add_argument("--checkpoint_filepath", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda:1")
    return parser.parse_args()


def train_and_evaluate(
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    criterion_generator: torch.nn.Module,
    optimizer_generator: torch.optim,
    train_dl: torch.utils.data.DataLoader,
    eval_dl: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    log_interval: int = 20,
    eval_interval: int = 200,
    checkpoint_save_dirpath: str = "./checkpoints/",
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
            torch.cuda.empty_cache()
            global_step += 1
            generator.train()

            # batch data
            image, label = batch
            image, label = image.to(args.device), label.to(args.device)

            # forward generator
            reconstructed, logp_x, mean, std = generator(image, label)

            # backward generator
            (
                final_generator_loss,
                recon_loss,
                kl_divergence,
                cnf_log_prob,
            ) = criterion_generator(
                image_true=image,
                image_pred=reconstructed,
                mean=mean,
                std=std,
                logp_x=logp_x,
            )
            final_generator_loss.backward()
            generator_grad = utils.clip_and_get_grad_values(generator)
            optimizer_generator.step()
            optimizer_generator.zero_grad()

            # result of step
            step_pbar.set_description(
                f"[Global step: {global_step}, Generator loss: {final_generator_loss.item():.2f}]"
            )

            if global_step % log_interval == 0:
                # text logging
                _info = ""
                _info += f"\n=== Global step: {global_step} ==="
                _info += f"\nGradient"
                _info += f"\n\tgenerator_grad: {generator_grad.item():.2f}"
                _info += f"\nTraining Loss"
                _info += f"\n\tfinal_generator_loss: {final_generator_loss.item():.2f}"
                _info += f", recon_loss: {recon_loss.item():.2f}, kl_divergence: {kl_divergence.item():.2f}, cnf_log_prob: {cnf_log_prob.item():.2f}\n"
                if logger is not None:
                    logger.info(_info)

                # tensorboard logging
                scalar_dict = {}
                scalar_dict.update({"grad/generator_grad": generator_grad})
                scalar_dict.update({"final_generator_loss": final_generator_loss})
                scalar_dict.update({"final_generator_loss/recon_loss": recon_loss})
                scalar_dict.update({"final_generator_loss/kl_divergence": kl_divergence})
                scalar_dict.update({"final_generator_loss/cnf_log_prob": cnf_log_prob})
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
            reconstructed, logp_x, mean, std = generator(image, label)

            (
                eval_final_generator_loss,
                eval_recon_loss,
                eval_kl_divergence,
                eval_cnf_log_prob,
            ) = criterion_generator(
                image_true=image,
                image_pred=reconstructed,
                mean=mean,
                std=std,
                logp_x=logp_x,
            )
            break

        # text logging
        _info = ""
        _info += f"\n=== Global step: {global_step} ==="
        _info += f"\nEvaluation Loss"
        _info += f"\n\tfinal_generator_loss: {eval_final_generator_loss.item():.2f}"
        _info += f", recon_loss: {eval_recon_loss.item():.2f}, kl_divergence: {eval_kl_divergence.item():.2f}, cnf_log_prob: {eval_cnf_log_prob.item():.2f}\n"
        if logger is not None:
            logger.info(_info)

        # tensorboard logging
        scalar_dict = {}
        scalar_dict.update({"final_generator_loss": eval_final_generator_loss})
        scalar_dict.update({"final_generator_loss/recon_loss": eval_recon_loss})
        scalar_dict.update({"final_generator_loss/kl_divergence": eval_kl_divergence})
        scalar_dict.update({"final_generator_loss/cnf_log_prob": eval_cnf_log_prob})
        if tensorabord_eval_writer is not None:
            for k, v in scalar_dict.items():
                tensorabord_eval_writer.add_scalar(k, v, global_step)

        # save checkpoint
        generator_checkpoint_filepath = os.path.join(checkpoint_save_dirpath, f"cpt_gen_{global_step}.pth")
        utils.save_checkpoint(
            generator, optimizer_generator, epoch, global_step, generator_checkpoint_filepath, logger
        )

        # visualization
        if viz:
            condition = label[:1]
            z_t_samples, time_space = generator.generate(condition, n_viz_time_steps)
            condition = condition[0].cpu().item()
            utils.visualize_inference_result(z_t_samples, condition, time_space, viz_save_dirpath, global_step)

    epoch_pbar = tqdm(range(n_epochs))
    for _ in epoch_pbar:
        epoch += 1
        epoch_pbar.set_description(f"Epoch: {epoch}")
        train_and_evaluate_one_epoch()


def main(args):
    utils.seed_everything(args.seed)

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

    generator = VAECNF(
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

    # for p in generator.image_encoder.parameters():
    #     p.requires_grad = False
    # for p in generator.image_decoder.parameters():
    #     p.requires_grad = False
    # for p in generator.ode_func.parameters():
    #     p.requires_grad = False

    generator_n_params = utils.count_parameters(generator)
    logger.info(f"generator_n_params: {generator_n_params}")

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)

    if args.checkpoint_filepath:
        epoch, global_step, generator, optimizer_generator = utils.load_checkpoint(
            args.checkpoint_filepath, generator, optimizer_generator, logger
        )
    else:
        epoch, global_step = -1, -1

    criterion_generator = losses.FinalGeneratorLoss(
        disc_fake_feature_map_loss_weigth=args.disc_fake_feature_map_loss_weigth,
        gan_generator_loss_weight=args.gan_generator_loss_weight,
        recon_loss_weight=args.recon_loss_weight,
        kl_divergence_weight=args.kl_divergence_weight,
        cnf_loss_weight=args.cnf_loss_weight,
        return_only_final_loss=False,
    )

    train_and_evaluate(
        epoch=epoch,
        global_step=global_step,
        generator=generator,
        criterion_generator=criterion_generator,
        optimizer_generator=optimizer_generator,
        train_dl=train_dl,
        eval_dl=train_dl,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        viz=args.viz,
        n_viz_time_steps=args.n_viz_time_steps,
        checkpoint_save_dirpath=os.path.join(args.log_dirpath, "checkpoints"),
        viz_save_dirpath=os.path.join(args.log_dirpath, "viz"),
        logger=logger,
        tensorabord_train_writer=tensorboard_train_writer,
        tensorabord_eval_writer=tensorboard_eval_writer,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
