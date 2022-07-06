import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

from helper import read_config
from augmentation import HighPassFilter
from models import Encoder

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def knn_classify(k, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
   
    """
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc

parser = argparse.ArgumentParser(description='RandomMask')
parser.add_argument('config_paths', nargs='+')
args = parser.parse_args()
config = read_config(args.config_paths)

transform = transforms.Compose([
    transforms.ToTensor(),
    HighPassFilter(kernel_size=13, sigma=5.0),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[0.08, 0.08, 0.08]),
])

trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = Encoder(config.backbone_type, config.planes_multipliers).cuda()
model.eval()

ckpt = torch.load(os.path.join(f"{config.checkpoint_dir}/{config.comment}", "checkpoint_20.pt"), map_location='cpu')
print(f'resuming from checkpoint 20')

encoder_state_dict = {}
for k, v in ckpt['model_state_dict'].items():
    name = k[7:] # remove module.backbone.
    encoder_state_dict[name] = v

print(model.load_state_dict(encoder_state_dict, strict=False))

all_p = []
all_label = []
for step, inputs in enumerate(trainloader):
    x, label = inputs
    x = x.cuda()
    label = label.cuda()
    p = torch.flatten(F.adaptive_avg_pool2d(model(x), (1, 1)), 1)

    all_p.append(p)
    all_label.append(label)

train_features = torch.cat(all_p)
train_labels = torch.cat(all_label)

all_p = []
all_label = []
for step, inputs in enumerate(testloader):
    x, label = inputs
    x = x.cuda()
    label = label.cuda()
    p = torch.flatten(F.adaptive_avg_pool2d(model(x), (1, 1)), 1)

    all_p.append(p)
    all_label.append(label)

test_features = torch.cat(all_p)
test_labels = torch.cat(all_label)

knn_classify(config.k, train_features, train_labels, test_features, test_labels)

