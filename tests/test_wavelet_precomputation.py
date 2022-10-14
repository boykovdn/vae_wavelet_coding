import torch
from pytorch_wavelets import DWTForward
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import (
        Compose,
        CenterCrop,
        PILToTensor,
        ConvertImageDtype)

from vae_wavelet_coding.transforms import wavelet_transform_reshape
from vae_wavelet_coding.dataset import MNIST_wavelets
from pathlib import Path
import pytest

precomputation_dir = Path(__file__).parents[1] / Path("MNIST_p")
raw_dataset_dir = Path(__file__).parents[1]

@pytest.fixture
def mnist_p():

    pre_transforms = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float32)
        ])

    post_transforms = Compose([
            CenterCrop(16),
        ])

    wavelet_tr = lambda img : wavelet_transform_reshape(
            img, 
            DWTForward(J=1, mode='zero', wave='db2'))

    dataset = MNIST_wavelets(
            precomputation_dir,
            load_all = True,
            wavelet_transform = wavelet_tr,
            mnist_raw_transform = pre_transforms,
            mnist_after_transform = post_transforms,
            root = raw_dataset_dir, # root for raw MNIST dataset
            download = True,
            train = True
            )

    return dataset

def test_dataset_wavelet_targets(mnist_p):

    # Get directly from lists
    wavelet_0 = mnist_p.wavelets[0]
    target_0 = mnist_p.targets[0]

    # Use __getitem__
    wavelet_1, target_1 = mnist_p[0]

    # Switch to loading the wavelets from disk.
    mnist_p.load_all = False
    wavelet_2, target_2 = mnist_p[0]

    assert torch.all(wavelet_0 == wavelet_1)
    assert torch.all(wavelet_1 == wavelet_2)
    assert target_0 == target_1
    assert target_1 == target_2

    print(wavelet_0.shape, target_0)

    # Check that the images actually match the targets
    #####
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,5)
    for idx in range(5):
        wavelet, target = mnist_p[idx]
        axes[idx].imshow(wavelet[0].relu())
        axes[idx].set_title(target)

    plt.show()
    #####
