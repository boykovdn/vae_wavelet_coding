import torch
from tqdm import tqdm
from utils import get_dataset, get_dataloader
from loggers import logging_wavelets_visualization
from supn.ipe_vae import IPE_autoencoder_mu_l
from model import WaveletVAE
from trainers.ipe_vae import IPEVAETrainer, WaveletVAETrainer
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTInverse
from transforms import OddReLU

def main():

    ENCODING_DIM = 64 #config['ENCODING_DIM']
    DEPTH = 3 #config['DEPTH']
    batch_size = 256
    init_num_channels = 32
    tb_output_dir = "./tb_test_hh_test"
    training_type = "vae_full" # TODO This shouldn't matter if not using SUPN, keep to full.
    init_state_path = "./checkpoint_hh.model" # TODO For pdb testing of output ranges.
    save_state_path = "./checkpoint_hh_test.model"
    image_logging_period = 100
    max_iterations = 100000
    learning_rate = 1e-4
    STDEV = 0.05
    l1_weight = 1.
    use_rescaling = True # Whether to force all channels to have similar scale during training.
    use_residual_blocks = True
    odd_relu_eps = None #0.1
    device=0

    summary_writer = SummaryWriter(log_dir=tb_output_dir)

    dataloader = get_dataloader(batch_size=batch_size)
    input_shape = dataloader.dataset[0][0].shape # (1, 128, 128)

    if odd_relu_eps is not None:
        odd_relu_factory = lambda : \
                OddReLU(eps=odd_relu_eps, trainable=False, 
                        device="cuda:{}".format(device))
    else:
        odd_relu_factory = None

    model = WaveletVAE(
                input_shape, 
                ENCODING_DIM, 
                dim_h=ENCODING_DIM, 
                depth=DEPTH,
                init_num_channels=init_num_channels,
                final_mu_activation = odd_relu_factory,
                output_channels=1, # The number of wavelet parameters per location.
                use_residual_blocks=use_residual_blocks
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
                    summary_writer,
                    use_rescaling=use_rescaling)

    trainer = WaveletVAETrainer(model, 
            summary_writer=summary_writer,
            learning_rate=learning_rate,
            stdev=STDEV,
            use_rescaling=use_rescaling,
            l1_weight=l1_weight)

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
