import torch
import torch.nn as nn

from .modules import Conv, SPPF, C2f

class Backbone(nn.Module):
  def __init__(
    self, 
    base_channels,
    base_depth
  ):
    super(Backbone, self).__init__()
    self.stem = Conv(3, base_channels, 3, 2)
    
    self.dark2 = nn.Sequential(
      Conv(
        base_channels,
        base_channels * 2,
        kernel_size=3,
        stride=2
      ),
      C2f(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        shortcut=True
      ),
    )
    self.dark3 = nn.Sequential(
      Conv(
        base_channels * 2,
        base_channels * 4,
        kernel_size=3, 
        stride=2
      ),
      C2f(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 2,
        shortcut=True
      )
    )
    self.dark4 = nn.Sequential(
      Conv(
        base_channels * 4,
        base_channels * 8,
        kernel_size=3, 
        stride=2
      ),
      C2f(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 2,
        shortcut=True)
    )
    self.dark5 = nn.Sequential(
      Conv(
        base_channels * 8,
        base_channels * 16,
        kernel_size=3, 
        stride=2
      ),
      C2f(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=True
      ),
      SPPF(
        base_channels * 16,
        base_channels * 16,
        kernel_size=5
      )
    )
    
  def forward(self, x):
    x = self.stem(x)
    x = self.dark2(x)
    x     = self.dark3(x)
    feat1 = x
    x     = self.dark4(x)
    feat2 = x
    x     = self.dark5(x)
    feat3 = x
    return feat1, feat2, feat3
