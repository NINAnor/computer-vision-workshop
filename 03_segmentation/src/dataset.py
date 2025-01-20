import glob

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        albumentations_transform=None,
        resize_transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.albumentations_transform = albumentations_transform
        self.resize_transform = resize_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open images as PIL Images
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Apply torchvision resizing if provided
        if self.resize_transform is not None:
            img = self.resize_transform(img)
            mask = self.resize_transform(mask)

        # Convert to numpy arrays for albumentations
        img = np.array(img)
        mask = np.array(mask)

        # Apply albumentations transformations
        if self.albumentations_transform is not None:
            augmented = self.albumentations_transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

            # Convert img and mask to torch tensors
        img = img.clone().detach().float()
        mask = mask.clone().detach().long()
        
        return img, mask


def get_data_loaders(
    image_dir,
    mask_dir,
    albumentations_transform,
    resize_transform,
    batch_size=8,
    val_split=0.2,
    num_workers=8,
):
    image_paths = sorted(glob.glob(image_dir + "/*.png"))
    mask_paths = sorted(glob.glob(mask_dir + "/*.tif"))

    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = (
        train_test_split(image_paths, mask_paths, test_size=val_split, random_state=42)
    )


    # Create datasets
    train_dataset = SegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        albumentations_transform=albumentations_transform,
        resize_transform=resize_transform,
    )

    val_dataset = SegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        albumentations_transform=albumentations_transform,
        resize_transform=resize_transform,
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
