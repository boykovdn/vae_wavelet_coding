import os
from datetime import datetime
import argparse
import yaml
import logging
from pathlib import Path
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, MNIST
from torchvision.transforms import (
        Compose,
        CenterCrop,
        PILToTensor,
        Grayscale,
        Normalize,
        ConvertImageDtype)
from supn.utils import rescale_to

from vae_wavelet_coding.dataset import MNIST_wavelets
from vae_wavelet_coding.transforms import wavelet_transform_reshape
from pytorch_wavelets import DWTForward

class RescalingModule(torch.nn.Module):
    r"""
    Scales the mean and variance to 1 of the wavelet channels (all but 0th).
    It is a part of the model, and the scaling parameters are only set once,
    the first time this transform is applied.
    """

    def __init__(self):

        super().__init__()

        #self.mu = None
        self.var = None

        # Flag that determines whether the mu and variance parameters will be
        # updated on the next forward pass of the transform.
        self.update_parameters = True

    def apply_scaling(self, x):
        r"""
        Args:
            :x: [B,C,H,W] torch.Tensor

        Returns:
            [B,C,H,W] torch.Tensor, mean 0 and var 1 each channel.
        """
        if self.update_parameters:
            # Initialize the mean and var.
            #self.mu = x.mean((0,2,3))[None,][...,None,None].to(x.device) # [1, C, 1,1]
            self.var = x.abs().mean((0,2,3))[None,][...,None,None].to(x.device) # [1, C, 1,1]
            self.var[:,0] = 1. # Ensure that the low-low channel is not scaled.

            #self.mu.requires_grad = False
            self.var.requires_grad = False

            self.update_parameters = False

        #out_ = (x - self.mu) / self.std
        out_ = x / self.var

        return out_

    def invert_scaling(self, x):
        r"""
        Args:
            :x: [B,C,H,W] mean 0 and var 1 each channel

        Returns:
            [B,C,H,W] rescaled and mean added back.
        """
        #return (x * self.std) + self.mu
        return x * self.var

    def __call__(self, x):

        return self.apply_scaling(x)

def get_dataset(split="train", dataset_name="mnist", load_into_mem=True, 
        precompute_path = None, wavelet_type='db2'):
    r"""
    Handles loading the dataset object and adding the relevant transforms to it.

    Args:
        :split: str, train/test/valid/all, as required by the CelebA class from
            PyTorch.

        :dataset_name: str, identifier for which dataset to use.

        :load_into_mem: bool, if True will load the datapoints into RAM if the
            dataset class allows it. This parameter is exposed in case I ever
            need to work with datasets too large for memory (which is not 
            expected).

        :precompute_path: str or Path, precomputation directory, if None will
            use the current directory.
    """
    allowed_splits = ["train", "test"]

    assert split in allowed_splits, "Expected split in {}, but got {}"\
            .format(allowed_splits, split)

    if dataset_name == "celeba":

        dataset = CelebA(
                root=Path(__file__).parents[1],
                download=True,
                split=split
                )

        dataset.transform = Compose([
                PILToTensor(),
                CenterCrop(128),
                Grayscale(),
                ConvertImageDtype(torch.float32),
                Normalize(0. ,1.)
            ])

    elif dataset_name == "mnist":

        train_split = split == "train"

        wavelet_tr = lambda img : wavelet_transform_reshape(
                img, 
                DWTForward(J=1, mode='zero', wave=wavelet_type))

        pre_transforms = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float32)
        ])

        post_transforms = Compose([
            CenterCrop(16),
        ])

        if load_into_mem:
            print("Loading dataset into memory!")

        if precompute_path is None:
            precompute_path = "./"

        dataset = MNIST_wavelets(
                Path(precompute_path) / Path(
                    "MNIST_p_{}".format(split)),
                root = Path(precompute_path), # root for raw MNIST dataset
                load_all = load_into_mem,
                wavelet_transform = wavelet_tr,
                mnist_raw_transform = pre_transforms,
                mnist_after_transform = post_transforms,
                download = True,
                train = train_split
            )

    else:
        raise Exception(
                "Dataset identifier {} not recognised.".format(dataset_name)
                )

    return dataset

def get_dataloader(batch_size=64, split="train", precompute_path=None,
        wavelet_type='db2'):
    r"""
    By default expecting the celeba dataset to be located at the root
    of the directory, else it will try to download it, possibly unsuccessfully
    due to the dataset being hosted in Google Drive with a daily quota.

    Dataset transforms make the images center-cropped 128x128 size and
    variance-normalized to mean 0, var 1, single-precision, grayscale.
    """
    dataset = get_dataset(split=split, precompute_path=precompute_path,
            wavelet_type=wavelet_type)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def setup_logging(logging_path):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.FileHandler(logging_path, mode='w')
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_timestamp_string():
    dtime = datetime.now()
    outp_ = "{}_{:02d}_{}_{}{}{}" \
            .format(dtime.year,
                    dtime.month,
                    dtime.day,
                    dtime.hour,
                    dtime.minute,
                    dtime.second)

    return outp_

def load_config_file(config_filepath):
    """
    Helper function that reads a yaml file and returns its contents as a dict.
    Args:
        :param config_filepath: str, a path pointing to the yaml config.
    """
    with open(config_filepath, "r") as yaml_config:
        yaml_dict = yaml.load(yaml_config, Loader=yaml.Loader)
        return yaml_dict

def parse_config_dict(description, config_arg_help):
    """
    Helper function which requires the user to submit a yaml config file before 
    running the rest of the code following it. It will then load the contents 
    of the config file and return them as a dict. 
    
    Passing a single yaml config file will be needed in a couple of places 
    throughout the algorithm (training and inference).
    Args:
        :param description: str, the program description that will be shown to
            the user.
    Returns:
        :argparse.ArgumentParser: Will prompt the user for a --config argument, 
            followed by a path to a .yaml configuration file.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help=config_arg_help, required=True)
    args = parser.parse_args()

    return load_config_file(args.config)

def make_dirs_if_absent(DIR_LIST):
    for DIR in DIR_LIST:
        if not os.path.exists(DIR):
            print("creating dir {}".format(DIR))
            os.makedirs(DIR)
