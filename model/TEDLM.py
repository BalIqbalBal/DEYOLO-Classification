import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.decomposition import PCA

# VGGFace Feature Extractor
class VGGFaceFeatureExtractor(nn.Module):
    def __init__(self, input_channel):
        super(VGGFaceFeatureExtractor, self).__init__()

        # Use VGG16-like architecture
        self.vgg16 = models.vgg16(pretrained=False)  
        self.vgg16.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        
        # Retain convolutional layers and intermediate fully connected layers
        self.features = self.vgg16.features
        self.fc_layers = nn.Sequential(*list(self.vgg16.classifier[:-1]))  # Exclude last layer

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)     
        return x
    
class VGGFaceFeatureExtractor_image(nn.Module):
    def __init__(self, input_channel):
        super(VGGFaceFeatureExtractor, self).__init__()

        # Use VGG16-like architecture
        self.vgg16 = models.vgg16(pretrained=False)
        
        # Modify the first layer to accept the specified number of input channels
        self.vgg16.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        
        # Retain only the convolutional layers (exclude the fully connected layers)
        self.features = self.vgg16.features

    def forward(self, x):
        # Extract features using convolutional layers
        x = self.features(x)  # Output shape: [batch_size, 512, 7, 7]
        
        # Return the feature map directly without flattening (shape: [batch_size, 512, 7, 7])
        return x

# Linear PCA
class LinearPCA:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit(self, features):
        self.pca.fit(features)

    def transform(self, features):
        return self.pca.transform(features)

    def fit_transform(self, features):
        return self.pca.fit_transform(features)

# DNN1
class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5))
        self.bilstm = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # Apply adaptive pooling to reduce channels to 1 before passing to conv1
        x = x.mean(dim=1, keepdim=True)  # Reduce channels to 1

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), x.size(2), -1)  # Flatten to (batch_size, seq_len, features)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        return x[:, -1, :]  # Return the last output

# DNN2
class DNN2(nn.Module):
    def __init__(self):
        super(DNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5))
        self.bilstm = nn.LSTM(input_size=128, hidden_size=32, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), x.size(2), -1)  # Flatten to (batch_size, seq_len, features)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        return x[:, -1, :]  # Return the last output

# DNN3
class DNN3(nn.Module):
    def __init__(self):
        super(DNN3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5)
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x.squeeze(1)))  # Conv1d expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # Swap dimensions to (batch_size, seq_len, features)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        return x[:, -1, :]  # Return the last output

# Late Fusion Model
class LateFusionModel(nn.Module):
    def __init__(self):
        super(LateFusionModel, self).__init__()
        self.dnn1 = DNN1()
        self.dnn2 = DNN2()
        self.dnn3 = DNN3()

        # Initialize fc as a placeholder, to be updated later
        self.fc = None

    def forward(self, x):
        # Forward pass through each DNN to get their outputs
        out1 = self.dnn1(x)
        out2 = self.dnn2(x)
        out3 = self.dnn3(x)

        # Concatenate the outputs from all DNNs
        fused = torch.cat((out1, out2, out3), dim=1)

        # Print the shape of the fused tensor for debugging
        #print(f"Shape of fused tensor: {fused.shape}")

        # Initialize the fully connected layer only once
        if self.fc is None:
            # Set the input size as the second dimension of fused (number of features)
            input_size = fused.size(1)  # Number of features after concatenation
            self.fc = nn.Linear(input_size, 5)  # Map to 5 output classes

        # Pass the fused tensor through the fully connected layer
        output = self.fc(fused)
        return output

class TEDLMFeatureFusion(nn.Module):
    def __init__(self, n_components=15):
        super(TEDLMFeatureFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(3)
        self.pca = LinearPCA(n_components=n_components)
        self.late_fusion = LateFusionModel()
        
        # Optionally, store fitted PCA during training or pre-processing
        self.pca_fitted = False

    def forward(self, rgb, thermal):
        # Feature extraction for RGB and thermal
        rgb_features = self.vggface(rgb)
        thermal_features = self.vggface(thermal)

        # Concatenate RGB and thermal features
        combined_features = torch.cat((rgb_features, thermal_features), dim=1)

        # If PCA has been fitted, transform using the fitted PCA model
        if self.pca_fitted:
            combined_features_np = combined_features.detach().cpu().numpy()
            reduced_features = torch.tensor(self.pca.transform(combined_features_np)).to(combined_features.device)
        else:
            reduced_features = combined_features  # Use raw features if PCA hasn't been applied

        # Reshape for DNN input (assumes 1 channel input for each DNN)
        dnn_input = reduced_features.unsqueeze(1).unsqueeze(2)

        # Late fusion DNN
        output = self.late_fusion(dnn_input)
        output = F.softmax(output, dim=1)
        return output

class TEDLMStackFusion(nn.Module):
    def __init__(self, n_components):
        super(TEDLMStackFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(4)
        self.pca = LinearPCA(n_components=n_components)
        self.late_fusion = LateFusionModel()

    def forward(self, rgb_image, thermal_image):
        # Early fusion: stack RGB and thermal images
        fused_input = torch.cat((rgb_image, thermal_image), dim=1)  # Stack along channel axis
        # Extract VGGFace features
        features = self.vggface(fused_input)
        # PCA dimensionality reduction (requires offline PCA fitting)
        features_pca = torch.tensor(self.pca.transform(features.detach().numpy()), dtype=torch.float32)
        # Reshape PCA features for DNNs
        features_pca = features_pca.unsqueeze(1).unsqueeze(-1)  # Add channel dimensions for DNNs
        # Late fusion
        output = self.late_fusion(features_pca)
        return output


class TEDLMStackFusionWithoutPCA(nn.Module):
    def __init__(self):
        super(TEDLMStackFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(4)  # Assuming 4 channels for fused input (RGB + thermal)
        self.late_fusion = LateFusionModel()

    def forward(self, rgb_image, thermal_image):
        # Early fusion: stack RGB and thermal images along the channel axis
        fused_input = torch.cat((rgb_image, thermal_image), dim=1)  # Stack along channel axis

        # Extract features from VGGFace
        features = self.vggface(fused_input)

        # Reshape features for DNN input
        features_pca = features.unsqueeze(1).unsqueeze(-1)  # Add channel dimensions for DNNs

        # Late fusion
        output = self.late_fusion(features_pca)
        return output

class TEDLMFeatureFusionWithoutPCA(nn.Module):
    def __init__(self):
        super(TEDLMFeatureFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(3)  # Assuming 3 channels (RGB)
        self.late_fusion = LateFusionModel()  # Assuming the late fusion model handles final output
    
    def forward(self, rgb, thermal):
        # Feature extraction for RGB and thermal
        rgb_features = self.vggface(rgb)
        thermal_features = self.vggface(thermal)

        # Concatenate RGB and thermal features along the channel dimension
        combined_features = torch.cat((rgb_features, thermal_features), dim=1)

        # Reshape for DNN input (assumes 1 channel input for each DNN)
        dnn_input = combined_features.unsqueeze(1).unsqueeze(2)

        # Late fusion DNN
        output = self.late_fusion(dnn_input)
        output = F.softmax(output, dim=1)
        return output

class StackRGBThermalVGGFaceDNN1(nn.Module):
    def __init__(self):
        super(StackRGBThermalVGGFaceDNN1, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(6)  # RGB + Thermal stacked input (channels=6)
        self.dnn1 = DNN1()  # Using DNN1 as the final classifier

    def forward(self, rgb_image, thermal_image):
        # Ensure the images are in the correct shape (batch_size, channels, height, width)
        # Stack RGB and thermal images along the channel axis
        fused_input = torch.cat((rgb_image, thermal_image), dim=1)  # Stack RGB and thermal along channel axis

        # Check if the shape is [batch_size, channels, height, width] before passing to VGGFace
        print(f"Shape of fused input: {fused_input.shape}")

        # Extract features using VGGFace
        features = self.vggface(fused_input)

        # Pass the features through DNN1
        output = self.dnn1(features)
        return output