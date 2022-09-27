import torch
from tqdm import tqdm
from utils import get_dataset, get_dataloader
from loggers import logging_wavelets_visualization
from supn.ipe_vae import IPE_autoencoder_mu_l
from model import WaveletVAE
from trainers.ipe_vae import IPEVAETrainer, WaveletVAETrainer
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTInverse

def main():

    ENCODING_DIM = 32 #config['ENCODING_DIM']
    DEPTH = 3 #config['DEPTH']
    batch_size = 256
    init_num_channels = 16
    tb_output_dir = "./tb_test_1"
    training_type = "mean_only"
    init_state_path = None #"./checkpoint_mean.model"
    save_state_path = "./checkpoint.model"
    image_logging_period = 100
    max_iterations = 100000
    learning_rate = 1e-3
    STDEV = 0.05
    device=0

    summary_writer = SummaryWriter(log_dir=tb_output_dir)

    dataloader = get_dataloader(batch_size=batch_size)
    input_shape = dataloader.dataset[0][0].shape # (1, 128, 128)

    model = WaveletVAE(
                input_shape, 
                ENCODING_DIM, 
                dim_h=ENCODING_DIM, 
                depth=DEPTH,
                init_num_channels=init_num_channels,
                output_channels=4 # The number of wavelet parameters per location.
            ).to(device)

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
                    summary_writer)

    trainer = WaveletVAETrainer(model, 
            summary_writer=summary_writer,
            learning_rate=learning_rate,
            stdev=STDEV)

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
