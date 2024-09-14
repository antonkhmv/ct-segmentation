from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.transforms as tvt

from models import Unet3d, Unet2d

parser = ArgumentParser()
parser.add_argument("-c", "--checkpoint", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-o", "--output", type=str, default="model.pt", required=False)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)

model_type = args.model
if model_type == "unet3d":
    params = checkpoint["trial"].params
    model = Unet3d(**params)
    random_input = torch.randn(1, 1, 256, 256, 256, dtype=torch.float32)
elif model_type == "unet2d":
    params = checkpoint["trial"].params
    model = Unet2d(**params)
    random_input = torch.randn(1, 1, 512, 512, dtype=torch.float32)
else:
    raise ValueError()


class Normalize(nn.Module):
    def forward(self, image):
        xmin, xmax = image.min(), image.max()
        image = (image - xmin + 1) / (xmax - xmin + 1)
        return image


model.load_state_dict(checkpoint["model"])

model = nn.Sequential(
    tvt.Resize((128, 128), interpolation=tvt.InterpolationMode.NEAREST),
    Normalize(),
    model,
    tvt.Resize((512, 512), interpolation=tvt.InterpolationMode.NEAREST)
)
model.eval()

with torch.no_grad():
    traced = torch.jit.trace(model, random_input)

torch.jit.save(traced, args.output)
