import torch
from tqdm import tqdm
from utils import get_dataset, get_dataloader
from loggers import logging_wavelets_visualization
from supn.ipe_vae import IPE_autoencoder_mu_l
from model import WaveletVAE
from trainers.ipe_vae import IPEVAETrainer, WaveletVAETrainer
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTInverse

import wandb

def main():

    ENCODING_DIM = 64 #config['ENCODING_DIM']
    DEPTH = 3 #config['DEPTH']
    batch_size = 256
    init_num_channels = 32
    training_type = "vae_full" # TODO This shouldn't matter if not using SUPN, keep to full.
    init_state_path = "./checkpoint_resid.model" # TODO For pdb testing of output ranges.
    save_state_path = "./checkpoint_resid.model"
    image_logging_period = 100
    max_iterations = 100000
    learning_rate = 1e-4
    STDEV = 1.
    laplace_b = 1.
    kl_weight = 1.
    use_rescaling = True # Whether to force all channels to have similar scale during training.
    use_residual_blocks = True
    device=0

    wandb.init(
            project="test-supn",
            config={
                "batch_size" : batch_size,
                "learning_rage" : learning_rate,
                "gauss_std" : STDEV,
                "laplace_b" : laplace_b,
                "kl_weight" : kl_weight
                }
            )

    dataloader = get_dataloader(batch_size=batch_size)
    input_shape = dataloader.dataset[0][0].shape # (1, 128, 128)

    model = WaveletVAE(
                input_shape, 
                ENCODING_DIM, 
                dim_h=ENCODING_DIM, 
                depth=DEPTH,
                init_num_channels=init_num_channels,
                output_channels=4, # The number of wavelet parameters per location.
                use_residual_blocks=use_residual_blocks
            ).to(device)

    wandb.watch(model)

    # Logging
    # Datapoints unseen during the training process.
    dset_test = get_dataset(split="test")

    def image_logging_function(it):
        if it % image_logging_period == 0:
            ifm = DWTInverse(mode='zero', wave='db2').to(device)
            logging_wavelets_visualization(
                    model,
                    dset_test,
                    ifm,
                    it,
                    wandb,
                    use_rescaling=use_rescaling)

    trainer = WaveletVAETrainer(model, 
            summary_writer=wandb,
            learning_rate=learning_rate,
            use_rescaling=use_rescaling,
            stdev=STDEV,
            laplace_b=laplace_b,
            kl_weight=kl_weight)

    trainer.train(
                dataloader, 
                save_state_path,
                image_logging=image_logging_function, 
                device=0, 
                max_iterations=max_iterations,
                init_state_path=init_state_path
            )

if __name__ == "__main__":
    main()
