import torch
import torch.nn as nn

from .utils import compute_padding

class Conv(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size=1,
    stride=1,
    padding=None,
    groups=1,
    dilation=1
  ):
    super(Conv, self).__init__()
    padding = compute_padding(kernel_size, padding=padding, dilation=dilation)
    self.conv = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size,
      stride=stride,
      padding=padding,
      groups=groups,
      dilation=dilation,
      bias=False
    )
    self.bn = nn.BatchNorm2d(out_channels)
    self.act = nn.SiLU()

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

class SPPF(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=5):
    super(SPPF, self).__init__()
    self.cv1 = Conv(
      in_channels,
      in_channels // 2,
      kernel_size=1,
      stride=1
    )
    self.cv2 = Conv(
      in_channels * 2,
      out_channels,
      kernel_size=1,
      stride=1
    )
    self.m = nn.MaxPool2d(
      kernel_size=kernel_size,
      stride=1,
      padding=kernel_size // 2
    )

  def forward(self, x):
    y = [self.cv1(x)]
    y.extend(self.m(y[-1]) for _ in range(3))
    return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    shortcut=True,
    groups=1,
    kernel_size=(3, 3),
    e=0.5
  ):
    super(Bottleneck, self).__init__()
    self.cv1 = Conv(
      in_channels,
      int(out_channels * e),
      kernel_size=kernel_size[0],
      stride=1
    )
    self.cv2 = Conv(
      int(out_channels * e),
      out_channels,
      kernel_size[1],
      stride=1, 
      groups=groups
    )
    self.residual = shortcut and (in_channels == out_channels)

  def forward(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C2f(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    n=1,
    shortcut=False,
    groups=1,
    e=0.5
  ):
    super(C2f, self).__init__()
    self.out_channels = out_channels
    self.e = e
    
    self.cv1 = Conv(
      in_channels,
      2 * int(self.out_channels * self.e),
      kernel_size=1,
      stride=1
    )
    self.cv2 = Conv(
      (2 + n) * int(self.out_channels * self.e),
      out_channels,
      kernel_size=1
    )
    self.m = nn.ModuleList(
      Bottleneck(
        int(self.out_channels * self.e),
        int(self.out_channels * self.e),
        shortcut=shortcut,
        groups=groups, 
        kernel_size=((3, 3), (3, 3)),
        e=1.0
      )
        for _ in range(n)
    )

  def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))

  def forward_split(self, x):
    y = list(
      self.cv1(x).split(
        split_size_or_sections=(int(self.out_channels * self.e), int(self.out_channels * self.e)),
        dim=1)
    )
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
