from vae_wavelet_coding.utils import get_dataloader
from vae_wavelet_coding.model import WaveletVAE
from vae_wavelet_coding.trainers.ipe_vae import WaveletVAETrainer

import pytest

ENCODING_DIM = 64
DEPTH = 3
init_num_channels = 32
use_residual_blocks = True
use_rescaling = True
batch_size = 128
max_iterations = 1000
device = "cuda:0"

@pytest.fixture
def dataloader():

    return get_dataloader(batch_size=batch_size)

def test_profiling(dataloader):

    input_shape = dataloader.dataset[0][0].shape

    model = WaveletVAE(
            input_shape,
            ENCODING_DIM,
            dim_h=ENCODING_DIM,
            depth=DEPTH,
            init_num_channels=init_num_channels,
            output_channels=4, # The number of wavelet parameters per location.
            use_residual_blocks=use_residual_blocks
        ).to(device)

    trainer = WaveletVAETrainer(model,
        summary_writer=None,
        learning_rate=1e-5,
        stdev=1.,
        use_rescaling=use_rescaling,
        l1_weight=1.)

    trainer.train(
            dataloader,
            None,
            device=0,
            max_iterations=max_iterations)

if __name__ == "__main__":
    # If script is run by the profiler.
    dataloader = get_dataloader(batch_size=batch_size)
    test_profiling(dataloader)
