"""
VGG code and pre-trained weigths from https://github.com/huyvnphan/PyTorch_CIFAR10
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_cifar10_mu_std_img
from dataset import normalize
from path import *

__all__ = [
    "VGG",
    "vgg11_bn",
]

# vgg architecture definition for CIFAR 10
class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    # extract feature at a specific "layer"
    def extract_features(self, x, layer):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if layer == "last_conv":
            return x
        elif layer == "last_fc":
            last_layers = self.classifier[:-1]
            x = last_layers(x)
            return x
        else:
            raise NotImplementedError
            

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# "M": MaxPool2d
# "A": vgg11
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}

# load a vgg model
def _vgg(arch, cfg, batch_norm, pretrained, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = torch.load(
            os.path.join(model_path, arch + ".pt"), map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(pretrained=False, device="cpu", **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar10
    """
    return _vgg("vgg11_bn", "A", True, pretrained, device, **kwargs)

def test_pretrained_vgg(x_test, y_test):
    vgg_network = vgg11_bn(pretrained=True)
    vgg_network.eval()

    # normalize image dataset
    mu_img, std_img = get_cifar10_mu_std_img()
    x_test_ = normalize(np.copy(x_test).astype(np.float32), mu_img, std_img)

    batch_size = 100 
    num_test_samples = x_test.shape[0]

    num_test_batches = num_test_samples // batch_size
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(num_test_batches):
            
            # forward NN
            output_batch = vgg_network(
                torch.tensor(x_test_[i*batch_size :(i+1)*batch_size]))
            
            _, predicted_labels = torch.max(output_batch, 1)

            # get ground truth label tensor 
            gt_labels = y_test[i*batch_size :(i+1)*batch_size]
            gt_labels = torch.from_numpy(gt_labels)
            

            total += gt_labels.size(0)
            correct += (predicted_labels == gt_labels).sum().item()

    test_acc = correct / total
    
    return test_acc
    
    
    
