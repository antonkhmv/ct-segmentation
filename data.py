import os
from abc import ABC
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile
import torch
from nibabel.dft import pydicom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, items, transforms, image_key, mask_key, load_image):
        self.transforms = transforms
        self.image_key = image_key
        self.mask_key = mask_key
        self.load_image = load_image
        self.items = items

    def __getitem__(self, index) -> torch.tensor:
        paths = self.items[index]
        image = self.load_image(paths[self.image_key])
        mask = self.load_image(paths[self.mask_key])
        image, mask = self.transforms(image, mask)
        return image, mask

    def __len__(self):
        return len(self.items)


class AbstractLoader(ABC):
    @staticmethod
    def load_3d_image(image_path: str) -> torch.Tensor:
        """
        Load numpy array of pixels from image file.
        :param image_path: path to file
        :return: 3d array of pixels, H x W x L
        """
        raise NotImplementedError()

    @staticmethod
    def load_2d_image(image_path):
        """
        Load torch tensor of pixels from saved file.
        :param image_path: path to file
        :return: 2d array of pixels, H x W
        """
        raise NotImplementedError()

    @staticmethod
    def save_3d_image(image_path: str, image: torch.Tensor):
        """
        Save torch tensor of pixels.
        :param image: tensor of pixels
        :param image_path: path to file
        """
        raise NotImplementedError()

    @staticmethod
    def save_2d_image(image_path: str, image: torch.Tensor):
        """
        Save torch tensor of pixels.
        :param image: tensor of pixels
        :param image_path: path to file
        """
        raise NotImplementedError()


class NIfTILoader(AbstractLoader):
    @staticmethod
    def load_3d_image(image_path):
        """
        Load numpy array of pixels from NIfTi file.
        :param image_path: path to file
        :return: 3d array of pixels, H x W x L
        """
        image = nib.load(image_path).get_fdata()
        image = np.rot90(image).copy()
        image = torch.from_numpy(image).transpose(0, 2)
        return image


class DICOMLoader(AbstractLoader):
    @staticmethod
    def load_2d_image(image_path):
        """
        Load torch tensor of pixels from saved file.
        :param image_path: path to file
        :return: 2d array of pixels, H x W
        """
        return torch.from_numpy(np.asarray(pydicom.dcmread(image_path, force=True).pixel_array).astype(float))

    @staticmethod
    def load_2d_image_float(image_path):
        """
        Load torch tensor of pixels from saved file.
        :param image_path: path to file
        :return: 2d array of pixels, H x W
        """
        return DICOMLoader.load_2d_image(image_path) / 4096

    @staticmethod
    def save_2d_image_float(image_path: str, image: torch.Tensor):
        """
        Save torch tensor of pixels.
        :param image: tensor of pixels
        :param image_path: path to file
        """
        tifffile.imwrite(image_path, (image * 4096).long().numpy(), photometric='minisblack')
        return image_path

    @staticmethod
    def load_3d_image(image_path):
        """
        Load numpy array of pixels from DICOM file.
        :param image_path: path to file
        :return: 3d array of pixels, H x W x L
        """
        slices = [DICOMLoader.load_2d_image(str(file)) for file in sorted(Path(image_path).iterdir())]
        return torch.cat(slices, dim=0)


class TIFFLoader(AbstractLoader):
    @staticmethod
    def load_2d_image(image_path):
        """
        Load torch tensor of pixels from saved file.
        :param image_path: path to file
        :return: 2d array of pixels, H x W
        """
        image = tifffile.imread(image_path)
        return torch.from_numpy(image * 1.0).float()

    @staticmethod
    def load_2d_image_float(image_path):
        """
        Load torch tensor of pixels from saved file.
        :param image_path: path to file
        :return: 2d array of pixels, H x W
        """
        image = tifffile.imread(image_path)
        return torch.from_numpy(image * 1.0).float() / 4096

    @staticmethod
    def save_2d_image(image_path: str, image: torch.Tensor):
        """
        Save torch tensor of pixels.
        :param image: tensor of pixels
        :param image_path: path to file
        """
        tifffile.imwrite(image_path, image.long().numpy(), photometric='minisblack')
        return image_path

    @staticmethod
    def save_2d_image_float(image_path: str, image: torch.Tensor):
        """
        Save torch tensor of pixels.
        :param image: tensor of pixels
        :param image_path: path to file
        """
        tifffile.imwrite(image_path, (image * 4096).long().numpy(), photometric='minisblack')
        return image_path


def get_loader(loader_name) -> AbstractLoader:
    if loader_name == "NIfTILoader":
        return NIfTILoader()
    elif loader_name == "DICOMLoader":
        return DICOMLoader()
    elif loader_name == "TIFFLoader":
        return TIFFLoader()
    else:
        raise ValueError(f"Unknown loader name {loader_name}")


def collate_fn(tensors):
    return [torch.cat(batch, dim=0).unsqueeze(1).float() for batch in list(zip(*tensors))]


def make_loaders(config, items, transforms, image_key, mask_key, loader_func):
    train, test = train_test_split(items, test_size=config.test_size, random_state=config.random_state, shuffle=True)
    ret = []
    for name, split in [("train", train), ("test", test)]:
        ret.append(
            DataLoader(
                SegmentationDataset(split, transforms[name], image_key, mask_key, loader_func),
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
            )
        )
    return ret


def read_manifest(data_dir):
    manifest = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(manifest)
    return [row for _, row in df.iterrows()]


def download_kaggle_dataset(dataset, path):
    import kaggle

    kaggle.api.dataset_download_files(dataset=dataset, path=path, unzip=True, quiet=False)
