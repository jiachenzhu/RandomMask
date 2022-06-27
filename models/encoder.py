import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

class Encoder(nn.Module):
    def __init__(self, backbone_type, planes_multipliers):
        super().__init__()

        if backbone_type == 'resnet':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64 * planes_multipliers[0], kernel_size=8, stride=4, bias=False),
                nn.BatchNorm2d(64 * planes_multipliers[0]),
                nn.ReLU(),
                nn.Conv2d(64 * planes_multipliers[0], 64 * planes_multipliers[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(64 * planes_multipliers[1]),
                nn.ReLU(),
                nn.Conv2d(64 * planes_multipliers[1], 64 * planes_multipliers[2], kernel_size=1, bias=True),
            )
        elif backbone_type == 'convnext':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                LayerNorm2d(96, eps=1e-6),
                # nn.Conv2d(96, 96, kernel_size=7, padding=3, groups=96, bias=True),
                # Permute([0, 2, 3, 1]),
                # nn.LayerNorm(96, eps=1e-6),
                # nn.Linear(in_features=96, out_features=4 * 96, bias=True),
                # nn.GELU(),
                # nn.Linear(in_features=4 * 96, out_features=96, bias=True),
                # Permute([0, 3, 1, 2]),
                nn.GELU(),
                nn.Conv2d(96, 96 * planes_multipliers[0], kernel_size=8, stride=4, bias=False),
                LayerNorm2d(96 * planes_multipliers[0], eps=1e-6),
                nn.GELU(),
                nn.Conv2d(96 * planes_multipliers[0], 96 * planes_multipliers[1], kernel_size=1, bias=False),
                LayerNorm2d(96 * planes_multipliers[1]),
                nn.GELU(),
                nn.Conv2d(96 * planes_multipliers[1], 96 * planes_multipliers[2], kernel_size=1, bias=True),
            )
  
    def forward(self, x):
        return self.backbone(x)
