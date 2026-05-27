'''ResNet in PyTorch for TinyImageNet.

Adapted from the original ResNet implementation for TinyImageNet dataset.
Key changes:
- Optimized for 64x64 input images (vs 32x32 CIFAR)
- Default 200 classes for TinyImageNet
- Adjusted pooling layer for proper feature map dimensions

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # First conv layer - optimized for 64x64 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final classifier
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: 64x64x3
        out = F.relu(self.bn1(self.conv1(x)))  # 64x64x64
        out = self.layer1(out)                 # 64x64x64
        out = self.layer2(out)                 # 32x32x128
        out = self.layer3(out)                 # 16x16x256
        out = self.layer4(out)                 # 8x8x512
        
        # Global average pooling for 8x8 feature maps
        out = F.avg_pool2d(out, 8)             # 1x1x512
        out = out.view(out.size(0), -1)        # 512
        out = self.linear(out)                 # num_classes
        return out


def ResNet18_TinyImageNet():
    """ResNet18 for TinyImageNet (200 classes, 64x64 images)"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200)


def ResNet50_TinyImageNet():
    """ResNet50 for TinyImageNet (200 classes, 64x64 images)"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200)


def test():
    """Test function to verify model output dimensions"""
    print("Testing ResNet18_TinyImageNet:")
    net18 = ResNet18_TinyImageNet()
    y = net18(torch.randn(1, 3, 64, 64))
    print(f"Input: (1, 3, 64, 64) -> Output: {y.size()}")
    
    print("\nTesting ResNet50_TinyImageNet:")
    net50 = ResNet50_TinyImageNet()
    y = net50(torch.randn(1, 3, 64, 64))
    print(f"Input: (1, 3, 64, 64) -> Output: {y.size()}")
    
    # Parameter count
    print(f"\nResNet18 parameters: {sum(p.numel() for p in net18.parameters() if p.requires_grad):,}")
    print(f"ResNet50 parameters: {sum(p.numel() for p in net50.parameters() if p.requires_grad):,}")


if __name__ == '__main__':
    test()