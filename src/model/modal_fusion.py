from .DEYOLO import DEA
import torch.nn as nn
from torchvision import models
import torch
from .common import Conv

from .vgg_face import VGGFace


class resnetDecaDepaHead(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(resnetDecaDepaHead, self).__init__()

        # Define DEA module for attention (only for layers [9, 19])
        self.dea = DEA(2048, 20)  # For layers [9, 19]

        # Classification head
        c1, c2 = 2048, 5  # Input channels for DEA output, output classes
        self.conv = Conv(c1, 1280, 1, 1)  # EfficientNet-b0 size
        self.pool = nn.AdaptiveAvgPool2d(1)  # Pool to shape (b, c_, 1, 1)
        self.drop = nn.Dropout(p=dropout_rate, inplace=True)  # Dropout with the specified rate
        self.linear = nn.Linear(1280, c2)  # Final fully connected layer

    def forward(self, backbone_outputs):
        """
        Forward pass of the SimpleHead module.

        Args:
            backbone_outputs (list): Combined outputs from RGB and thermal backbones.

        Returns:
            torch.Tensor: Classifier output.
        """
        # Apply DEA module (only for layers [9, 19])
        attention = self.dea((backbone_outputs[0], backbone_outputs[1]))  # Layer pair [9, 19]

        # Pass through the classification head
        x = self.conv(attention)  # Shape: (batch, 1280, 20, 20)
        x = self.pool(x)  # Shape: (batch, 1280, 1, 1)
        x = x.flatten(1)  # Shape: (batch, 1280)
        x = self.drop(x)  # Apply dropout
        x = self.linear(x)

        return x

class vggfacveDecaDepaHead(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(vggfacveDecaDepaHead, self).__init__()

        # Define DEA module for attention (only for layers [9, 19])
        self.dea = DEA(25088, 20)  # For layers [9, 19]

        # Classification head
        c1, c2 = 25088, 5  # Input channels for DEA output, output classes
        self.conv = Conv(c1, 1280, 1, 1)  # EfficientNet-b0 size
        self.pool = nn.AdaptiveAvgPool2d(1)  # Pool to shape (b, c_, 1, 1)
        self.drop = nn.Dropout(p=dropout_rate, inplace=True)  # Dropout with the specified rate
        self.linear = nn.Linear(1280, c2)  # Final fully connected layer

    def forward(self, backbone_outputs):
        """
        Forward pass of the SimpleHead module.

        Args:
            backbone_outputs (list): Combined outputs from RGB and thermal backbones.

        Returns:
            torch.Tensor: Classifier output.
        """
        # Apply DEA module (only for layers [9, 19])
        attention = self.dea((backbone_outputs[0], backbone_outputs[1]))  # Layer pair [9, 19]

        # Pass through the classification head
        x = self.conv(attention)  # Shape: (batch, 1280, 20, 20)
        x = self.pool(x)  # Shape: (batch, 1280, 1, 1)
        x = x.flatten(1)  # Shape: (batch, 1280)
        x = self.drop(x)  # Apply dropout
        x = self.linear(x)

        return x

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
        self.head = resnetDecaDepaHead(dropout_rate=dropout_rate)

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

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)  # Extract features from ResNet

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
    def __init__(self, dropout_rate=0.2, rgb_weights_path="./weights/vgg_face_dag.pth", thermal_weights_path=None):
        super(vggfaceDecaDepa, self).__init__()

        # Define the VGG-Face backbone for RGB input
        self.rgb_backbone = VGGFace()
        if rgb_weights_path:
            # Load the pretrained VGG-Face weights
            state_dict = torch.load(rgb_weights_path)
            self.rgb_backbone.load_state_dict(state_dict)

        # Define the VGG-Face backbone for Thermal input (optional)
        self.thermal_backbone = VGGFace()
        if thermal_weights_path:
            # Load custom pretrained weights for thermal input (if provided)
            state_dict = torch.load(thermal_weights_path)
            self.thermal_backbone.load_state_dict(state_dict)

        # Remove the classifier to use the backbone as a feature extractor
        self.rgb_backbone.fc6 = nn.Identity()
        self.rgb_backbone.fc7 = nn.Identity()
        self.rgb_backbone.fc8 = nn.Identity()

        self.thermal_backbone.fc6 = nn.Identity()
        self.thermal_backbone.fc7 = nn.Identity()
        self.thermal_backbone.fc8 = nn.Identity()

        # Define dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)

        # Define the simplified head module (only uses DEA for layers [9, 19])
        self.head = vggfacveDecaDepaHead(dropout_rate=dropout_rate)

    def forward(self, rgb_image, thermal_image):
        # Forward pass through the RGB backbone
        rgb_features = self.rgb_backbone(rgb_image)  # Use the full forward pass of VGGFace
        rgb_features = self.dropout(rgb_features)

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)  # Use the full forward pass of VGGFace
        thermal_features = self.dropout(thermal_features)

        # Combine the features from both modalities (Thermal and RGB)
        combined_features = [
            rgb_features.unsqueeze(-1).unsqueeze(-1),  # Reshape to (batch, channels, 1, 1)
            thermal_features.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch, channels, 1, 1)
        ]

        # Forward pass through the simplified head with the combined features
        output = self.head(combined_features)
        return output