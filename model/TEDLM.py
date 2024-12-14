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
        self.fc = nn.Linear(256 + 64 + 256, 5)  # Sum of DNN outputs

    def forward(self, x):
        out1 = self.dnn1(x)
        out2 = self.dnn2(x)
        out3 = self.dnn3(x)
        fused = torch.cat((out1, out2, out3), dim=1)  # Concatenate outputs
        output = self.fc(fused)  # Map to 5 classes
        return output

# Fusion Model
class TEDLMFeatureFusion(nn.Module):
    def __init__(self, n_components=128):
        super(TEDLMFeatureFusion, self).__init__()
        self.vggface = VGGFaceFeatureExtractor(3)
        self.pca = LinearPCA(n_components=n_components)
        self.late_fusion = LateFusionModel()

    def forward(self, rgb, thermal):
        # Feature extraction for RGB and thermal
        rgb_features = self.vggface(rgb)
        thermal_features = self.vggface(thermal)

        # Concatenate RGB and thermal features
        combined_features = torch.cat((rgb_features, thermal_features), dim=1)

        # Apply PCA (convert to numpy for PCA processing)
        combined_features_np = combined_features.detach().cpu().numpy()
        reduced_features = torch.tensor(self.pca.transform(combined_features_np)).to(combined_features.device)

        # Reshape for DNN input (assumes 1 channel input for each DNN)
        dnn_input = reduced_features.unsqueeze(1).unsqueeze(2)

        # Late fusion DNN
        output = self.late_fusion(dnn_input)
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
