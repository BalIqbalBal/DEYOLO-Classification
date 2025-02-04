import os

from PIL import Image
from torchvision import transforms

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import numpy as np

def getDualImageDataloader(rgb_dir, thermal_dir, rgb_transform=None, thermal_transform=None, batch_size=32, test_size=0.2, shuffle=True, num_workers=4):
    """
    Creates and returns train and test DataLoader instances for the DualImageDataset.

    Args:
        rgb_dir (str): Directory path for RGB (VIS) images.
        thermal_dir (str): Directory path for thermal (NIR) images.
        rgb_transform (callable, optional): Optional transforms for RGB images.
        thermal_transform (callable, optional): Optional transforms for thermal images.
        batch_size (int): Batch size for DataLoader.
        test_size (float): Proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
        shuffle (bool): Whether to shuffle the data before splitting.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
    """

    rgb_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    thermal_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the full dataset
    dataset = DualImageDataset(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        rgb_transform=rgb_transform,
        thermal_transform=thermal_transform
    )

    # Calculate sizes for train and test splits
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size

    # Split the dataset into train and test subsets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader instances for train and test subsets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle the test data
        num_workers=num_workers
    )

    return train_loader, test_loader


class DualImageDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, rgb_transform=None, thermal_transform=None):
        """
        Args:
            rgb_dir (str): Directory path for RGB (VIS) images.
            thermal_dir (str): Directory path for thermal (NIR) images.
            rgb_transform (callable, optional): Optional transforms for RGB images.
            thermal_transform (callable, optional): Optional transforms for thermal images.
        """
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.rgb_transform = rgb_transform
        self.thermal_transform = thermal_transform

        self.data = self._load_data()

    def _load_data(self):
        """Load and pair RGB (VIS) and thermal (NIR) image paths based on filenames."""
        data = []
        for label_name in os.listdir(self.rgb_dir):
            rgb_label_dir = os.path.join(self.rgb_dir, label_name)
            thermal_label_dir = os.path.join(self.thermal_dir, label_name)

            if not (os.path.isdir(rgb_label_dir) and os.path.isdir(thermal_label_dir)):
                continue

            label = int(label_name.replace('label', ''))
            rgb_files = set(os.listdir(rgb_label_dir))  # For quick lookups
            thermal_files = os.listdir(thermal_label_dir)

            # Pair thermal (NIR) files with their corresponding RGB (VIS) files
            for thermal_file in thermal_files:
                if '_n_' in thermal_file:
                    vis_file = thermal_file.replace('_n_', '_d_')
                    if vis_file in rgb_files:
                        rgb_path = os.path.join(rgb_label_dir, vis_file)
                        thermal_path = os.path.join(thermal_label_dir, thermal_file)
                        data.append((rgb_path, thermal_path, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, thermal_path, label = self.data[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        thermal_image = Image.open(thermal_path).convert("RGB")  # Adjust if thermal uses a different mode

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.thermal_transform:
            thermal_image = self.thermal_transform(thermal_image)

        return rgb_image, thermal_image, label