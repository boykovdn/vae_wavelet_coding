import pytest
import torch
from torchvision.datasets import CelebA
from torchvision.transforms import (
        Compose, 
        CenterCrop, 
        PILToTensor, 
        Grayscale,
        Normalize,
        ConvertImageDtype)
from torch.utils.data import DataLoader
from pytorch_wavelets import DWTForward, DWTInverse
from vae_wavelet_coding.transforms import (
        wavelet_transform_reshape,
        inverse_wavelet_transform)
from pathlib import Path
import numpy as np

@pytest.fixture
def celeba():

    dataset = CelebA(
            root=Path(__file__).parents[1], 
            download=True,
            )

    # To (B,C,*) shape
    dataset.transform = Compose([
            PILToTensor(),
            CenterCrop(128),
            Grayscale(),
            ConvertImageDtype(torch.float32),
            Normalize(0. ,1.)
        ]) 

    return dataset

@pytest.fixture
def celeba_dataloader(celeba):

    dataloader = DataLoader(celeba, batch_size=64)

    return dataloader

def test_draw_samples(celeba_dataloader):

    for img, target in celeba_dataloader:
        print(img.shape)
        print(target.shape)
        # Do something
        break

def test_wavelet_decomposition_application(celeba):
    r"""
    Apply the wavelet decomposition to an image from the dataset, apply the
    inverse, then compare.
    """
    # Forward transformation
    xfm = DWTForward(J=1, mode='zero', wave='db2')
    img, _ = celeba[0]
    out_ = wavelet_transform_reshape(img, xfm)

    # Backward transformation
    ifm = DWTInverse(mode='zero', wave='db2')
    img_i = inverse_wavelet_transform(out_.unsqueeze(0), ifm)

    np.testing.assert_array_almost_equal(
            img.cpu().numpy()[None,], img_i.cpu().numpy())
