import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from helper import AverageMeter

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class VICRegLossModule:
    def __init__(self):
        self.loss_meter = AverageMeter()

    def __call__(self, projection_1, projection_2, sim_coeff, std_coeff, cov_coeff):
        world_size = torch.distributed.get_world_size()
        
        x = projection_1
        y = projection_2

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        batch_size, num_features = x.shape
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

        self.loss_meter.update(loss.item())
        
        return loss

    def __str__(self):
        return  f"L/{self.loss_meter.avg:.4f}"

    def reset_meters(self):
        self.loss_meter.reset()
