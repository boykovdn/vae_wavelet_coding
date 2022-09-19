import torch
from tqdm import tqdm
from utils import get_dataset, get_dataloader
from loggers import logging_wavelets_visualization
from supn.ipe_vae import IPE_autoencoder_mu_l
from trainers.ipe_vae import IPEVAETrainer
from torch.utils.tensorboard import SummaryWriter

def main():

    ENCODING_DIM = 2 #config['ENCODING_DIM']
    DEPTH = 3 #config['DEPTH']
    batch_size = 64
    init_num_channels = 8
    tb_output_dir = "./tb_test"
    training_type = "mean_only"
    init_state_path = None #"./checkpoint_mean.model"
    save_state_path = "./checkpoint.model"
    image_logging_period = 100
    max_iterations = 100000
    learning_rate = 1e-3
    STDEV = 0.1
    device=0

    summary_writer = SummaryWriter(log_dir=tb_output_dir)

    dataloader = get_dataloader(batch_size=batch_size)
    input_shape = dataloader.dataset[0][0].shape # (1, 128, 128)

    model = IPE_autoencoder_mu_l(
                input_shape, 
                ENCODING_DIM, 
                dim_h=ENCODING_DIM, 
                depth=DEPTH,
                init_num_channels=init_num_channels
            ).to(device)

    # Logging
    # Datapoints unseen during the training process.
    dset_test = get_dataset(split="test")
    def image_logging_function(it):
        if it % image_logging_period == 0:
            logging_wavelets_visualization(
                    model,
                    dset_test,
                    it,
                    summary_writer)

    trainer = IPEVAETrainer(model, 
            training_type=training_type, 
            summary_writer = summary_writer,
            learning_rate = learning_rate,
            enforce_l2=True,
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
