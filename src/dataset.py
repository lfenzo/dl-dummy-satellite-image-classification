import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision.io import read_image


PIXEL_CHANNEL_MEANS = (0.4914, 0.4822, 0.4465)
PIXEL_CHANNEL_STDS = (0.2023, 0.1994, 0.2010)


def denormalize_images(
    images,
    means: list[float] = PIXEL_CHANNEL_MEANS,
    stds: list[float] = PIXEL_CHANNEL_STDS,
):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def get_train_transforms():
    return v2.Compose([
        v2.Resize(64),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=PIXEL_CHANNEL_MEANS, std=PIXEL_CHANNEL_STDS)
    ])


def get_eval_transforms():
    return v2.Compose([
        v2.Resize(64),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=PIXEL_CHANNEL_MEANS, std=PIXEL_CHANNEL_STDS),
    ])


class SatelliteDataset(Dataset):

    CLASSES = {
        'cloudy': 0,
        'desert': 1,
        'green_area': 2,
        'water': 3,
    }

    def __init__(self, file: str, transforms=None):
        self.data = pd.read_csv(file)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        x = read_image(self.data.loc[idx, "path"], mode="RGB")
        y = torch.tensor(self.CLASSES[self.data.loc[idx, "class"]])

        if self.transforms:
            x = self.transforms(x)

        return x, y
