import os
import requests
import zipfile
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from collections import Counter

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import numpy as np


# URL for dataset download (replace this with your actual dataset URL)
DATASET_URL = "https://pain-identification-datasets.s3.ap-southeast-1.amazonaws.com/formatted_dataset.rar"
DATASET_DIR = "dataset/formatted_dataset"

def download_and_extract(url, dest_dir):
    """
    Downloads and extracts the dataset if it doesn't exist.

    Args:
        url (str): URL to download the dataset from.
        dest_dir (str): Directory to save the extracted dataset.
    """
    print(f"Dataset not found. Downloading from {url}...")
    
    # Create destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    # Download the dataset
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a valid response
    
    # Extract zip file
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(dest_dir)

    print(f"Dataset downloaded and extracted to {dest_dir}")

def check_and_download_dataset():
    """
    Checks if the dataset exists. If not, it downloads and extracts it.
    """
    if not os.path.exists(DATASET_DIR):
        download_and_extract(DATASET_URL, DATASET_DIR)

def getSingleImageDataloader(batch_size=16, image_dir="dataset/formatted_dataset/rgb", image_type="rgb"):
    """
    Returns the train, validation, and test DataLoaders for a dataset containing either RGB or thermal images.
    Automatically balances the dataset using class weights and applies data augmentation to the training set.

    Args:
        batch_size (int): Batch size for the dataloaders.
        image_dir (str): Directory containing the images.
        image_type (str): Type of images ("rgb" or "thermal").

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),             # Convert the image to a tensor
    ])

    # Validation and test transformations (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    train_dataset = SingleImageDataset(image_dir=image_dir, image_type=image_type, transform=train_transform)
    val_test_dataset = SingleImageDataset(image_dir=image_dir, image_type=image_type, transform=val_test_transform)

    # Split dataset into training (80%), validation (10%), and testing (10%)
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Ensure the splits add up correctly
    if train_size + val_size + test_size != dataset_size:
        raise ValueError("Splits do not add up to the total dataset size.")

    train_dataset, val_dataset, test_dataset = random_split(val_test_dataset, [train_size, val_size, test_size])

    # Calculate class weights for the training dataset
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]  # Assuming the dataset returns (image, label)
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[labels[i]] for i in range(len(train_dataset))]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class SingleImageDataset(Dataset):
    def __init__(self, image_dir, image_type="rgb", transform=None):
        """
        Args:
            image_dir (str): Directory path for the images.
            image_type (str): Type of images ("rgb" or "thermal").
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.image_dir = image_dir
        self.image_type = image_type
        self.transform = transform

        # Collect all image file paths grouped by labels
        self.data = self._load_data()

    def _load_data(self):
        """Helper function to load image paths grouped by labels."""
        data = []
        for label_name in os.listdir(self.image_dir):  # Iterate over label directories
            label_dir = os.path.join(self.image_dir, label_name)

            if not os.path.isdir(label_dir):
                continue

            label = int(label_name.replace('label', ''))  # Extract label from directory name

            image_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

            for file_name in image_files:
                image_path = os.path.join(label_dir, file_name)
                data.append((image_path, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label)
        """
        image_path, label = self.data[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def getDualImageDataloader(batch_size=16, rgb_dir="dataset/formatted_dataset/rgb", thermal_dir="dataset/formatted_dataset/thermal", train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Returns the train, validation, and test DataLoaders for the LPFW dataset with RGB and thermal images.
    Automatically balances the dataset using class weights and applies data augmentation to the training set.

    Args:
        batch_size (int): Batch size for the dataloaders.
        rgb_dir (str): Directory containing RGB images.
        thermal_dir (str): Directory containing thermal images.
        train_split (float): Fraction of data to be used for training.
        val_split (float): Fraction of data to be used for validation.
        test_split (float): Fraction of data to be used for testing.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Define transformations
    train_rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert the image to a tensor
    ])

    train_thermal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Validation and test transformations (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    train_dataset = DualImageDataset(rgb_dir=rgb_dir, thermal_dir=thermal_dir, rgb_transform=train_rgb_transform, thermal_transform=train_thermal_transform)
    val_test_dataset = DualImageDataset(rgb_dir=rgb_dir, thermal_dir=thermal_dir, rgb_transform=val_test_transform, thermal_transform=val_test_transform)

    # Calculate sizes for train, validation, and test splits
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    test_size = dataset_size - train_size - val_size

    # Split dataset into training, validation, and testing
    train_dataset, val_dataset, test_dataset = random_split(val_test_dataset, [train_size, val_size, test_size])

    # Calculate class weights for the training dataset
    labels = [label for _, _, label in train_dataset]  # Extract labels from the training dataset
    class_counts = np.bincount(labels)  # Count occurrences of each class
    class_weights = 1. / class_counts  # Inverse of class counts
    samples_weights = class_weights[labels]  # Assign weights to each sample

    # Create WeightedRandomSampler
    weighted_sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True  # Allow sampling with replacement
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class DualImageDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, rgb_transform=None, thermal_transform=None):
        """
        Args:
            rgb_dir (str): Directory path for RGB images.
            thermal_dir (str): Directory path for thermal images.
            rgb_transform (callable, optional): Optional transforms to apply to RGB images.
            thermal_transform (callable, optional): Optional transforms to apply to thermal images.
        """
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.rgb_transform = rgb_transform
        self.thermal_transform = thermal_transform

        # Collect all image file paths grouped by labels
        self.data = self._load_data()

    def _load_data(self):
        """Helper function to load and pair RGB and thermal image paths grouped by labels"""
        data = []
        for label_name in os.listdir(self.rgb_dir):  # Iterate over label directories
            rgb_label_dir = os.path.join(self.rgb_dir, label_name)
            thermal_label_dir = os.path.join(self.thermal_dir, label_name)

            if not os.path.isdir(rgb_label_dir) or not os.path.isdir(thermal_label_dir):
                continue

            label = int(label_name.replace('label', ''))  # Extract label from directory name

            rgb_files = [f for f in os.listdir(rgb_label_dir) if os.path.isfile(os.path.join(rgb_label_dir, f))]
            thermal_files = [f for f in os.listdir(thermal_label_dir) if os.path.isfile(os.path.join(thermal_label_dir, f))]

            common_files = list(set(rgb_files) & set(thermal_files))
            common_files.sort()

            for file_name in common_files:
                rgb_path = os.path.join(rgb_label_dir, file_name)
                thermal_path = os.path.join(thermal_label_dir, file_name)
                data.append((rgb_path, thermal_path, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (rgb_image, thermal_image, label)
        """
        rgb_path, thermal_path, label = self.data[idx]

        # Load RGB and thermal images
        rgb_image = Image.open(rgb_path).convert("RGB")
        thermal_image = Image.open(thermal_path).convert("RGB")  # Assuming thermal images are stored as RGB

        # Apply transformations if provided
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.thermal_transform:
            thermal_image = self.thermal_transform(thermal_image)

        return rgb_image, thermal_image, label


