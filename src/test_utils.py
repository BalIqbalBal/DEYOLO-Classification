from torchvision import transforms
from torch.utils.data import DataLoader
from utils.datasets import DualImageDataset

def test_datasets_class():

    # Paths to the dataset directories
    rgb_dir = "dataset/formatted_dataset/rgb"
    thermal_dir = "dataset/formatted_dataset/thermal"

    # Define transformations (if any)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset and dataloader
    dataset = DualImageDataset(rgb_dir=rgb_dir, thermal_dir=thermal_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through the dataset
    for rgb_images, thermal_images, labels in dataloader:
        print(rgb_images.shape, thermal_images.shape, labels)

if __name__ == "__main__":
    test_datasets_class()