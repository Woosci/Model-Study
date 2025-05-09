import torch
import torch.nn as nn
class BasicBlock(nn.Module):
  
  expansion = 1

  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):

    super(BasicBlock, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.downsample = downsample

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # 잔차 연결을 할 때 x와 F(x)의 크기가 다르다면 다운샘플링 수행
    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out



class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super(Bottleneck, self).__init__()

    width = out_channels
    # 차원 수 줄이기

    self.conv1 = nn.Conv2d(in_channels, width, kernel_size = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(width)

    self.conv2 = nn.Conv2d(width, width, kernel_size = 3, stride = stride, padding = 1,bias = False)
    self.bn2 = nn.BatchNorm2d(width)

    self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size = 1, bias = False)
    self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

    self.relu = nn.ReLU(inplace = True)
    self.downsample = downsample

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

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    # 잔차 연결
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 1000):
    super(ResNet, self).__init__()
    self.in_channels = 64

    # 초기 레이어 (Conv + BN + ReLU + MaxPool)
    self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace = True)
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)


    # Residual Layers (BasicBlock or Bottleneck 사용)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)


    # Fully Connected Layer
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 가중치 초기화
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")

  def _make_layer(self, block, out_channels, blocks, stride = 1):
     # 어떤 블록(BasicBlock 또는 Bottleneck)을 몇 개(blocks 개수) 어떤 크기(out_channels)로 쌓아서 하나의 레이어로 만들자!"

    downsample = None

    # downsample이 필요한지 확인!!
    if stride != 1 or self.in_channels != out_channels * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size = 1, stride = stride, bias = False),
          nn.BatchNorm2d(out_channels * block.expansion)
          )



    layers = []

    layers.append(block(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.in_channels, out_channels))
      # 이후 2번째 블록부터는 stride 입력X -> default 값인 1적용 -> 크기 변화X

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x
  

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)