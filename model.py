import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import MultiheadAttention
from torchvision.models import resnet50, ResNet50_Weights

# CBAM Attention Module
class CBAMModule(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(CBAMModule, self).__init__()
        self.in_channels = in_channels

        # Channel Attention
        self.shared_dense_one = nn.Linear(in_channels, in_channels // ratio)
        self.shared_dense_two = nn.Linear(in_channels // ratio, in_channels)

        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        # Channel Attention
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)

        avg_pool_fc = self.shared_dense_two(F.relu(self.shared_dense_one(avg_pool.view(x.size(0), -1))))
        max_pool_fc = self.shared_dense_two(F.relu(self.shared_dense_one(max_pool.view(x.size(0), -1))))

        channel_attention = torch.sigmoid(avg_pool_fc + max_pool_fc).view(x.size(0), self.in_channels, 1, 1)
        x = x * channel_attention

        # Spatial Attention
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_attention = torch.sigmoid(self.conv(concat))

        return x * spatial_attention


# Vision Transformer Block
class VisionTransformer(nn.Module):
    def __init__(self, num_patches=196, projection_dim=64, transformer_layers=4):
        super(VisionTransformer, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=projection_dim, num_heads=4) for _ in range(transformer_layers)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(projection_dim, projection_dim) for _ in range(transformer_layers)
        ])

    def forward(self, x):
        for i in range(len(self.attention_layers)):
            x1 = F.layer_norm(x, [x.size(-1)])
            attention, _ = self.attention_layers[i](x1, x1, x1)
            x2 = x + attention
            x2 = F.layer_norm(x2, [x.size(-1)])
            x2 = F.relu(self.fc_layers[i](x2))
            x = x2 + x
        return x

# Final ResNet-CBAM-ViT Model
class HybridModel(nn.Module):
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        super(HybridModel, self).__init__()

        # ResNet50 Backbone - Without avgpool and fc
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # CBAM Attention
        self.cbam = CBAMModule(2048)

        # ViT Block
        self.flatten_dim = 2048
        self.vit_dense = nn.Linear(self.flatten_dim, 12544)  # 196 x 64
        self.vit = VisionTransformer(num_patches=196, projection_dim=64)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_dim + 12544, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)                   # [B, 2048, 7, 7]
        x = self.cbam(x)                       # [B, 2048, 7, 7]
        x_avg = self.avgpool(x)                # [B, 2048, 1, 1]
        x_resnet = x_avg.view(x_avg.size(0), -1)  # [B, 2048]


        # ViT Branch
        vit_input = self.vit_dense(x_resnet)   # [B, 12544]
        vit_input = vit_input.view(-1, 196, 64)
        vit_output = self.vit(vit_input)       # [B, 196, 64]
        vit_output = vit_output.flatten(start_dim=1)  # [B, 12544]

        # Classification Head
        combined = torch.cat((x_resnet, vit_output), dim=1)  # [B, 2048+12544]
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        output = self.fc2(combined)

        return output

# create model
def create_model(input_shape=(224, 224, 3), num_classes=4):
    return HybridModel(input_shape=input_shape, num_classes=num_classes)
