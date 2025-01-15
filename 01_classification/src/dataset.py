import os

from PIL import Image
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels for the images.
            transform (callable, optional): Transform to be applied to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_val_dataloaders(
    logger, data_dir, batch_size=16, val_split=0.2, num_workers=8
):
    """
    Splits the dataset into training and validation sets and returns DataLoaders.

    Args:
        data_dir (str): Path to the dataset root directory.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of the data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation sets.
    """
    class_to_idx = {"Cat": 0, "Dog": 1}
    image_paths = []
    labels = []

    # collect all image paths and their labels
    for label, idx in class_to_idx.items():
        folder_path = os.path.join(data_dir, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path):  # ensure the file exists
                image_paths.append(img_path)
                labels.append(idx)

    # take half the data for faster training
    image_paths = image_paths[: len(image_paths) // 2]
    labels = labels[: len(labels) // 2]
    # dataset statistics
    total_images = len(image_paths)
    class_counts = {label: labels.count(idx) for label, idx in class_to_idx.items()}

    # print statistics in a table
    stats_table = [
        ["Total Images", total_images],
        *[[label, count] for label, count in class_counts.items()],
    ]
    logger.info(
        f"\n{tabulate(stats_table, headers=['Class', 'Count'], tablefmt='grid')}"
    )

    # split the dataset into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=42
    )

    # define transforms/augmentations for both the training and validation sets
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PetDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = PetDataset(val_paths, val_labels, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
