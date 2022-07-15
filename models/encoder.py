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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, backbone_type, planes_multipliers):
        super().__init__()

        if backbone_type == 'resnet':
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Bottleneck(64, 64),
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
