from typing import Sized

import torch
import torch.nn as nn
import torchvision.transforms as T


class ComposeImageMask(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        if all(isinstance(transform, nn.Module) for transform in transforms):
            self.transforms = list(zip(transforms, [1.0] * len(transforms)))
        elif all(isinstance(transform, Sized) and len(transform) == 2 for transform in transforms):
            self.transforms = transforms
        else:
            raise ValueError("Transforms should be a list of nn.Module or pairs (nn.Module, probability)")

    def forward(self, image, mask):
        for transform, prob in self.transforms:
            if torch.rand(1) <= prob:
                image = transform(image, is_mask=False)
                mask = transform(mask, is_mask=True)
        return image, mask


class Resize3DImageMask(nn.Module):
    def __init__(self, size, interpolation, **params):
        super().__init__()
        assert len(size) == 3
        self.resize1 = T.Resize([size[1], size[2]], interpolation, **params)
        self.resize2 = T.Resize([size[0], size[2]], interpolation, **params)

    def forward(self, image, is_mask=False):
        dim1, dim2, dim3 = image.shape
        image = torch.cat([self.resize1(image[i, :, :].unsqueeze(0)) for i in range(dim1)], dim=0)
        image = torch.cat([self.resize2(image[:, i, :].unsqueeze(0)) for i in range(dim2)], dim=0)
        image = image.unsqueeze(0)
        return image


class Resize2DImageMask(nn.Module):
    def __init__(self, size, interpolation, **params):
        super().__init__()
        self.resize = T.Resize(size, interpolation, **params)

    def forward(self, image, is_mask):
        image = image.unsqueeze(0)
        image = self.resize(image)
        return image


class Normalize(nn.Module):
    def __init__(self, xmin=None, xmax=None):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax

    def forward(self, image, is_mask=False):
        if is_mask:
            return image
        xmin, xmax = image.min(), image.max()
        image = (image - xmin + 1) / (xmax - xmin + 1)
        return image
