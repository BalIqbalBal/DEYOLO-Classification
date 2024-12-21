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
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))
        self.bilstm = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
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
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
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
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.squeeze(2)
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

        self.fc = nn.Linear(832, 5)

    def forward(self, x):
        # Forward pass through each DNN to get their outputs
        out1 = self.dnn1(x)
        out2 = self.dnn2(x)
        out3 = self.dnn3(x)

        # Concatenate the outputs from all DNNs
        fused = torch.cat((out1, out2, out3), dim=1)

        # Pass the fused tensor through the fully connected layer
        output = self.fc(fused)
        return output

class FeatureReducer(nn.Module):
    def __init__(self, input_features, reduced_features):
        super(FeatureReducer, self).__init__()
        # Reduction path
        self.reducer = nn.Sequential(
            nn.Linear(input_features, 128),    # Hidden layer 1
            nn.BatchNorm1d(128),               # Batch normalization
            nn.ReLU(),
            nn.Linear(128, 64),                # Hidden layer 2
            nn.BatchNorm1d(64),                # Batch normalization
            nn.ReLU(),
            nn.Linear(64, reduced_features),   # Output layer
            nn.BatchNorm1d(reduced_features)   # Batch normalization
        )
        
        # Bottleneck linear layer for skip connection
        self.bottleneck = nn.Linear(input_features, reduced_features)

    def forward(self, x):
        # Skip connection
        reduced_features = self.reducer(x)
        bottleneck_features = self.bottleneck(x)
        
        # Combine outputs (skip connection with reduced features)
        return reduced_features + bottleneck_features

class TEDLMFeatureFusion(nn.Module):
    def __init__(self, n_components=15):
        super(TEDLMFeatureFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(3)
        self.CR = FeatureReducer(8192, 512)
        self.late_fusion = LateFusionModel()


    def forward(self, rgb, thermal):
        # Feature extraction for RGB and thermal
        rgb_features = self.vggface(rgb)
        thermal_features = self.vggface(thermal)

        # Concatenate RGB and thermal features
        combined_features = torch.cat((rgb_features, thermal_features), dim=1)

        # Channel Reduction
        reduced_features = self.CR(combined_features)  # Fit and reduce
        reduced_features = reduced_features.unsqueeze(2).unsqueeze(3)

        # Late fusion DNN
        output = self.late_fusion(reduced_features)
        output = F.softmax(output, dim=1)
        return output

class TEDLMStackFusion(nn.Module):
    def __init__(self, n_components):
        super(TEDLMStackFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(4)
        self.CR = FeatureReducer(4096, 512)
        self.late_fusion = LateFusionModel()

    def forward(self, rgb_image, thermal_image):
        # Early fusion: stack RGB and thermal images
        fused_input = torch.cat((rgb_image, thermal_image), dim=1)  # Stack along channel axis

        # Extract VGGFace features
        features = self.vggface(fused_input)

        # Channel Reduction
        reduced_features = self.CR(combined_features)  # Fit and reduce
        reduced_features = reduced_features.unsqueeze(2).unsqueeze(3)

        # Late fusion
        output = self.late_fusion(reduced_features)
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
        # Extract features using VGGFace
        features = self.vggface(fused_input)

        # Pass the features through DNN1
        output = self.dnn1(features)
        return output