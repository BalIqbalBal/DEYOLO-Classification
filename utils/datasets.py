import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def getSingleImageDataloader(batch_size=16, image_dir="dataset/formatted_dataset/rgb", test_split=0.2, image_type="rgb"):
    """
    Returns the train and test DataLoaders for a dataset containing either RGB or thermal images.

    Args:
        batch_size (int): Batch size for the dataloaders.
        image_dir (str): Directory containing the images.
        test_split (float): Fraction of data to be used for testing.
        image_type (str): Type of images ("rgb" or "thermal").

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """

    # Define transformations (if any)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    dataset = SingleImageDataset(image_dir=image_dir, image_type=image_type, transform=transform)

    # Split dataset into training and testing
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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


def getDualImageDataloader(batch_size=16, rgb_dir="dataset/formatted_dataset/rgb", thermal_dir="dataset/formatted_dataset/thermal", test_split=0.2):
    """
    Returns the train and test DataLoaders for the LPFW dataset with RGB and thermal images.
    
    Args:
        batch_size (int): Batch size for the dataloaders.
        rgb_dir (str): Directory containing RGB images.
        thermal_dir (str): Directory containing thermal images.
        test_split (float): Fraction of data to be used for testing.
    
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    
    # Define transformations (if any)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    dataset = DualImageDataset(rgb_dir=rgb_dir, thermal_dir=thermal_dir, transform=transform)

    # Split dataset into training and testing
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class DualImageDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, transform=None):
        """
        Args:
            rgb_dir (str): Directory path for RGB images.
            thermal_dir (str): Directory path for thermal images.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform

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
        if self.transform:
            rgb_image = self.transform(rgb_image)
            thermal_image = self.transform(thermal_image)

        return rgb_image, thermal_image, label


