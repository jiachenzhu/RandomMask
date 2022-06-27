import torch
from torchvision import transforms

class MaskGenerator:
    def __init__(self, blur_mask_transform, mask_percentage):
        self.blur_mask_transform = blur_mask_transform
        self.mask_percentage = mask_percentage

    def __call__(self, x):
        b, _, h, w = x.shape

        mask_1 = self.blur_mask_transform(torch.randn(b, 1, h, w))
        threshold_1 = torch.quantile(mask_1.view(b, -1), self.mask_percentage, dim=1, keepdim=True)
        mask_1 = (mask_1 > threshold_1.view(b, 1, 1, 1)).float()

        mask_2 = self.blur_mask_transform(torch.rand(b, 1, h, w))
        threshold_2 = torch.quantile(mask_2.view(b, -1), self.mask_percentage, dim=1, keepdim=True)
        mask_2 = (mask_2 > threshold_2.view(b, 1, 1, 1)).float()

        return mask_1.to(x.device), mask_2.to(x.device)

class HighPassFilter:
    def __init__(self, kernel_size, sigma):
        self.low_pass_filter = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, x):
        return x - self.low_pass_filter(x)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    HighPassFilter(kernel_size=13, sigma=5.0),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[0.08, 0.08, 0.08])
])
