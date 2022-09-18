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

def get_dataset(split="train", dataset_name="mnist"):
    r"""
    Handles loading the dataset object and adding the relevant transforms to it.

    Args:
        :split: str, train/test/valid/all, as required by the CelebA class from
            PyTorch.

        :dataset_name: str, identifier for which dataset to use.
    """
    allowed_splits = ["train", "test"]

    assert split in allowed_splits, "Expected split in {}, but got {}"\
            .format(allowed_splits, split)

    # TODO Add wavelet transformation
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

        dataset = MNIST(
                root=Path(__file__).parents[1],
                download=True,
                train=train_split)

        dataset.transform = Compose([
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(0. ,1.)
            ])

    else:
        raise Exception(
                "Dataset identifier {} not recognised.".format(dataset_name)
                )

    return dataset

def get_dataloader(batch_size=64, split="train"):
    r"""
    By default expecting the celeba dataset to be located at the root
    of the directory, else it will try to download it, possibly unsuccessfully
    due to the dataset being hosted in Google Drive with a daily quota.

    Dataset transforms make the images center-cropped 128x128 size and
    variance-normalized to mean 0, var 1, single-precision, grayscale.
    """
    dataset = get_dataset(split=split)

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
