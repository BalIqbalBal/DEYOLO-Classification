import torch
import torch.nn as nn

from .common import Conv, Bottleneck, C2f

class DEYOLOCLASS(nn.Module):
    def __init__(self):
        super(DEYOLOCLASS, self).__init__()

        # Define the backbone for both RGB and Thermal inputs
        self.rgb_backbone = DEYOLOBackbone()
        self.thermal_backbone = DEYOLOBackbone()

        # Define the head module
        self.head = Head()

    def forward(self, rgb_image, thermal_image):
        # Forward pass through the RGB backbone
        rgb_features = self.rgb_backbone(rgb_image)

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)

        # Combine the features from both modalities (Thermal and RGB)
        # Assuming the outputs from each backbone match in size, you can directly pair them
        combined_features = [
            (rgb_features[i], thermal_features[i]) for i in range(len(rgb_features))
        ]
        
        # Forward pass through the head with the combined features
        output = self.head(combined_features)
        return output


class SimpleDEYOLOCLASS(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SimpleDEYOLOCLASS, self).__init__()

        # Define the backbone for both RGB and Thermal inputs
        self.rgb_backbone = DEYOLOBackbone()
        self.thermal_backbone = DEYOLOBackbone()

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
        rgb_features = self.rgb_backbone(rgb_image)  # Returns outputs from layers [11, 18, 22]
        rgb_last_layer = rgb_features[-1]  # Extract the last layer's output (layer 22)

        # Forward pass through the Thermal backbone
        thermal_features = self.thermal_backbone(thermal_image)  # Returns outputs from layers [11, 18, 22]
        thermal_last_layer = thermal_features[-1]  # Extract the last layer's output (layer 22)

        # Apply dropout to the last layer's features
        rgb_last_layer = self.dropout(rgb_last_layer)
        thermal_last_layer = self.dropout(thermal_last_layer)

        # Combine the last layer's features from both modalities (Thermal and RGB)
        combined_features = [
            rgb_last_layer, thermal_last_layer  # Only the last layer's outputs are paired
        ]

        # Forward pass through the simplified head with the combined features
        output = self.head(combined_features)
        return output

class SimpleHead(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SimpleHead, self).__init__()

        # Define DEA module for attention (only for layers [9, 19])
        self.dea = DEA(1024, 20)  # For layers [9, 19]

        # Classification head
        c1, c2 = 1024, 5  # Input channels for DEA output, output classes
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

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        # Define DEA modules for attention
        self.dea_1 = DEA(256, 80)   # For layers [4, 14]
        self.dea_2 = DEA(512, 40)   # For layers [6, 16]
        self.dea_3 = DEA(1024, 20)  # For layers [9, 19]

        # Convolutional layers for upsampling
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)
        )
        self.upsample_3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)  # Already (7, 7), only adjust channels

        # Classification head
        c1, c2 = 1024 * 3, 5 # Updated input channels for concatenated features
        self.conv = Conv(c1, 1280, 1, 1)  # EfficientNet-b0 size
        self.pool = nn.AdaptiveAvgPool2d(1)  # Pool to shape (b, c_, 1, 1)
        self.drop = nn.Dropout(p=0.2, inplace=True)  # Optional dropout
        self.linear = nn.Linear(1280, c2)  # Final fully connected layer

    def forward(self, backbone_outputs):
        """
        Forward pass of the Head module.

        Args:
            backbone_outputs (list): Combined outputs from RGB and thermal backbones.

        Returns:
            tuple: Classifier output and intermediate attention outputs.
        """
        # Apply DEA modules
        attention_1 = self.dea_1((backbone_outputs[0][0], backbone_outputs[0][1]))  # Layer pair [4, 14]
        attention_2 = self.dea_2((backbone_outputs[1][0], backbone_outputs[1][1]))  # Layer pair [6, 16]
        attention_3 = self.dea_3((backbone_outputs[2][0], backbone_outputs[2][1]))  # Layer pair [9, 19]

        # Upsample attention outputs to the same size and channels
        upsampled_1 = self.upsample_1(attention_1)  # Shape: 1024x7x7
        upsampled_2 = self.upsample_2(attention_2)  # Shape: 1024x7x7
        upsampled_3 = self.upsample_3(attention_3)  # Shape: 1024x7x7

        # Concatenate upsampled features
        classifier_input = torch.cat([upsampled_1, upsampled_2, upsampled_3], dim=1)  # Shape: (batch, 1024*3, 7, 7)

        # Pass through the classification head
        x = self.linear(self.drop(self.pool(self.conv(classifier_input)).flatten(1)))
        #if self.training:
        return x #, (attention_1, attention_2, attention_3)
        #y = x.softmax(1)  # Final output
        #return y DONT USE SOFTMAX BECAUSE WE USE NN.CROSSENTROPY

class DEYOLOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv(3, 64, 3, 2),  # 0-P1/2
            Conv(64, 128, 3, 2),  # 1-P2/4
            C2f_BiFocus(128, 128),
            C2f_BiFocus(128, 128),
            C2f_BiFocus(128, 128),  # 2
            Conv(128, 256, 3, 2),  # 3-P3/8
            C2f(256, 256),
            C2f(256, 256),
            C2f(256, 256),
            C2f(256, 256),
            C2f(256, 256),
            C2f(256, 256),  # 4
            Conv(256, 512, 3, 2),  # 5-P4/16
            C2f(512, 512),
            C2f(512, 512),
            C2f(512, 512),
            C2f(512, 512),
            C2f(512, 512),
            C2f(512, 512),  # 6
            Conv(512, 1024, 3, 2),  # 7-P5/32
            C2f(1024, 1024),
            C2f(1024, 1024),
            C2f(1024, 1024)  # 8
        ])

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # Collect only the outputs of layers #4, #6, and #8
            if idx in {11, 18, 22}:
                outputs.append(x)
        return outputs

class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)

class DEA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """convolution pyramid"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(self.kernel_size/kernel1/kernel2), int(self.kernel_size/kernel1/kernel2), 0,
                      groups=channel),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else torch.mean(
                                                                                    glob_t, dim=[2, 3], keepdim=True)
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir


class DEPA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        """convolution merge"""
        if m_kernel is None:
            m_kernel = [3, 7]
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir

class C2f_BiFocus(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        self.bifocus = BiFocus(c2, c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))

        return self.bifocus(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BiFocus(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.focus_h = FocusH(c1, c1, 3, 1)
        self.focus_v = FocusV(c1, c1, 3, 1)
        self.depth_wise = DepthWiseConv(3 * c1, c2, 3)

    def forward(self, x):
        return self.depth_wise(torch.cat([x, self.focus_h(x), self.focus_v(x)], dim=1))


class FocusH(nn.Module):

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)

        x1[..., ::2, :], x1[..., 1::2, :] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2, :], x2[..., 1::2, :] = x[..., ::2, 1::2], x[..., 1::2, ::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2, :]
        result[..., 1::2, 1::2] = x1[..., 1::2, :]
        result[..., ::2, 1::2] = x2[..., ::2, :]
        result[..., 1::2, ::2] = x2[..., 1::2, :]

        return result


class FocusV(nn.Module):

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)

        x1[..., ::2], x1[..., 1::2] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2], x2[..., 1::2] = x[..., 1::2, ::2], x[..., ::2, 1::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2]
        result[..., 1::2, 1::2] = x1[..., 1::2]
        result[..., 1::2, ::2] = x2[..., ::2]
        result[..., ::2, 1::2] = x2[..., 1::2]

        return result


class DepthWiseConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = Conv(in_channel, in_channel, kernel, 1, 1, in_channel)
        self.point_conv = Conv(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)

        return out