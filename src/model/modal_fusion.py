from DEYOLO import SimpleHead
import torch.nn as nn
from torchvision import models
from vgg_face import VGG_16

class resnetDecaDepa(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(resnetDecaDepa, self).__init__()

        # Define the ResNet backbone for both RGB and Thermal inputs
        self.rgb_backbone = models.resnet50(pretrained=True)
        self.thermal_backbone = models.resnet50(pretrained=True)

        # Remove the fully connected layer from ResNet to use it as a feature extractor
        self.rgb_backbone.fc = nn.Identity()
        self.thermal_backbone.fc = nn.Identity()

        # Define dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)

        # Define the simplified head module (only uses DEA for layers [9, 19])
        self.head = SimpleHead(dropout_rate=dropout_rate)

    def forward(self, rgb_image, thermal_image):
        """
        Forward pass for the DEYOLOCLASS model.

        Args:
            rgb_image (torch.Tensor): Input RGB image tensor.
            thermal_image (torch.Tensor): Input thermal image tensor.

        Returns:
            torch.Tensor: Output logits from the classification head.
        """
        # Forward pass through the RGB backbone
        rgb_features = self.rgb_backbone(rgb_image)  # Extract features from ResNet
        rgb_features = rgb_features.view(rgb_features.size(0), -1)  # Flatten the features

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)  # Extract features from ResNet
        thermal_features = thermal_features.view(thermal_features.size(0), -1)  # Flatten the features

        # Apply dropout to the features
        rgb_features = self.dropout(rgb_features)
        thermal_features = self.dropout(thermal_features)

        # Combine the features from both modalities (Thermal and RGB)
        combined_features = [
            rgb_features.unsqueeze(-1).unsqueeze(-1),  # Reshape to (batch, channels, 1, 1)
            thermal_features.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch, channels, 1, 1)
        ]

        # Forward pass through the simplified head with the combined features
        output = self.head(combined_features)
        return output

class vggfaceDecaDepa(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(vggfaceDecaDepa, self).__init__()

        # Define the VGGFace backbone for both RGB and Thermal inputs
        self.rgb_backbone = VGG_16()  # Load VGGFace model
        self.thermal_backbone = VGG_16()  # Load VGGFace model

        # Remove the fully connected layers from VGGFace to use it as a feature extractor
        self.rgb_backbone.fc6 = nn.Identity()
        self.rgb_backbone.fc7 = nn.Identity()
        self.rgb_backbone.fc8 = nn.Identity()
        self.thermal_backbone.fc6 = nn.Identity()
        self.thermal_backbone.fc7 = nn.Identity()
        self.thermal_backbone.fc8 = nn.Identity()

        # Define dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)

        # Define the simplified head module (only uses DEA for layers [9, 19])
        self.head = SimpleHead(dropout_rate=dropout_rate)

    def forward(self, rgb_image, thermal_image):
        """
        Forward pass for the DEYOLOCLASS model.

        Args:
            rgb_image (torch.Tensor): Input RGB image tensor.
            thermal_image (torch.Tensor): Input thermal image tensor.

        Returns:
            torch.Tensor: Output logits from the classification head.
        """
        # Forward pass through the RGB backbone
        rgb_features = self.rgb_backbone(rgb_image)  # Extract features from VGGFace
        rgb_features = rgb_features.view(rgb_features.size(0), -1)  # Flatten the features

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)  # Extract features from VGGFace
        thermal_features = thermal_features.view(thermal_features.size(0), -1)  # Flatten the features

        # Apply dropout to the features
        rgb_features = self.dropout(rgb_features)
        thermal_features = self.dropout(thermal_features)

        # Combine the features from both modalities (Thermal and RGB)
        combined_features = [
            rgb_features.unsqueeze(-1).unsqueeze(-1),  # Reshape to (batch, channels, 1, 1)
            thermal_features.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch, channels, 1, 1)
        ]

        # Forward pass through the simplified head with the combined features
        output = self.head(combined_features)
        return output
