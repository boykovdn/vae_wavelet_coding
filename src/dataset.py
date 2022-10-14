import torch
import os
from tqdm import tqdm
from torchvision.datasets import MNIST

class MNIST_wavelets(MNIST):
    r"""
    Decorates the MNIST dataset with the functionality to precompute the 
    wavelet representations and save them on disk. When a datapoint is
    requested, the request is handled by loading from disk, or from RAM, which
    is faster than loading a raw MNIST digit and then applying the wavelet
    transformation on-the-fly.
    """

    def __init__(self, precompute_path, *args, load_all=False, 
            wavelet_transform=None, mnist_raw_transform=None,
            mnist_after_transform=None, ftag="wavelet", **kwargs):
        r"""
        Checks whether the precomputation folder exists, and if no, computes
        all wavelet representations and saves them to the folder.

        Inputs:

            :precompute_path: str, Path or None, path to the root folder where
                the computed wavelet transforms are stored.

            *args arguments passed to super class.

            :load_all: bool, default False. If True, we attempt to load the
                full dataset into memory in order to avoid the disk reads when
                sampling.

            :wavelet_transform: callable, applies the wavelet transform and 
                reshape to the input raw image.

            :mnist_raw_transform: callable, transform to be applied to MNIST
                images before the wavelet transform.

            :mnist_after_transform: callable, transform to be applied after the
                raw image has been wavelet transformed.

            :ftag: string, filename which is then followed by the index of the
                datapoint from the parent class.

            **kwargs passed to super class.
        """

        super().__init__(*args, transform=mnist_raw_transform, **kwargs)

        self.ftag = ftag
        self.load_all = load_all
        self.wavelet_transform = wavelet_transform
        self.precompute_path = precompute_path
        self.mnist_after_transform = mnist_after_transform

        # These are populated if the dataset is loaded into RAM.
        self.wavelets = None

        if not os.path.exists(precompute_path):

            os.makedirs(precompute_path) 

            for idx in tqdm(range(super().__len__()), 
                    desc="Transforming and saving to disk..."):

                image, _ = super().__getitem__(idx)

                image_wavelets = self.wavelet_transform(image)

                if self.mnist_after_transform is not None:
                    image_wavelets = self.mnist_after_transform(image_wavelets)

                torch.save(image_wavelets, "{}/{}_{:08d}.pt".format(
                    self.precompute_path,
                    self.ftag, 
                    idx))

        else:

            # Sanity-check the number of files and filenames.

            fnames_in_dir = os.listdir(precompute_path)

            assert len(fnames_in_dir) == super().__len__(),\
                    "Found {} files in dir {}, but expected {} datapoints"\
                    .format(len(fnames_in_dir), precompute_path, 
                            super().__len__())

            for fname in fnames_in_dir:
                assert ftag in fname and fname[-3:] == ".pt",\
                        "Unexpected filename {}".format(fname)

        if load_all:

            fnames = os.listdir(precompute_path)
            fnames.sort()

            wavelets = []
            for idx in tqdm(range(len(fnames)), 
                    desc="Loading wavelets into ram..."):

                fname = "{}_{:08d}.pt".format(self.ftag, idx)

                wavelet = torch.load("{}/{}".format(precompute_path, fname))

                wavelets.append(wavelet)

            self.wavelets = wavelets

    def __getitem__(self, idx):
            
        target = int(self.targets[idx])

        if self.load_all:

            wavelets = self.wavelets[idx]

        else:

            wavelets = torch.load("{}/{}_{:08d}.pt".format(
                    self.precompute_path,
                    self.ftag, 
                    idx))

        return wavelets, target
