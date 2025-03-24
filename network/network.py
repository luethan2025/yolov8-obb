import math

import torch
import torch.nn as nn

from .modules import Conv, C2f
from .backbone import Backbone
from .utils import make_anchors

class DFL(nn.Module):
  def __init__(self, c1):
    super(DFL, self).__init__()
    self.c1 = c1
    self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False).requires_grad_(False)
    x = torch.arange(c1, dtype=torch.float)
    self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))

  def forward(self, x):
    b, _, a = x.shape
    return self.conv(
      x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
    ).view(b, 4, a)

class Yolo(nn.Module):
  def __init__(
    self, 
    num_classes, 
    phi, 
    ne=1
  ):
    super(Yolo, self).__init__()
    depth_dict = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
    dep_mul = depth_dict[phi]

    width_dict = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    wid_mul = width_dict[phi]
    
    base_channels = int(wid_mul * 64)
    base_depth = max(round(dep_mul * 3), 1)
    self.backbone = Backbone(base_channels, base_depth)
    
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    self.conv3_for_upsample1 = C2f(
      base_channels * 16 + base_channels * 8,
      base_channels * 8,
      n=base_depth,
      shortcut=False
    )
    
    self.conv3_for_upsample2 = C2f(
      base_channels * 8 + base_channels * 4,
      base_channels * 4,
      n=base_depth,
      shortcut=False
    )
    
    self.down_sample1 = Conv(
      base_channels * 4,
      base_channels * 4,
      kernel_size=3,
      stride=2
    )

    self.conv3_for_downsample1 = C2f(
      base_channels * 8 + base_channels * 4,
      base_channels * 8,
      n=base_depth, 
      shortcut=False
    )

    self.down_sample2 = Conv(
      base_channels * 8,
      base_channels * 8,
      kernel_size=3,
      stride=2
    )

    self.conv3_for_downsample2 = C2f(
      base_channels * 16 + base_channels * 8,
      base_channels * 16,
      n=base_depth,
      shortcut=False
    )

    ch = [base_channels * 4, base_channels * 8, base_channels * 16]
    self.shape = None
    self.nc = num_classes
    self.nl = len(ch)
    self.reg_max = 16
    self.no = num_classes + self.reg_max * 4
    self.ne = ne

    self.stride = torch.tensor(
      [
        256 / x.shape[-2]
          for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))
      ]
    )
    c2 = max((16, ch[0] // 4, self.reg_max * 4))
    c3 = max(ch[0], self.nc)
    c4 = max(ch[0] // 4, self.ne)
    
    self.cv2 = nn.ModuleList(
      nn.Sequential(
        Conv(x, c2, kernel_size=3),
        Conv(c2, c2, kernel_size=3),
        nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1)
      ) for x in ch
    )

    self.cv3 = nn.ModuleList(
      nn.Sequential(
        Conv(x, c3, kernel_size=3),
        Conv(c3, c3, kernel_size=3),
        nn.Conv2d(c3, self.nc, kernel_size=1)
      ) for x in ch
    )

    self.cv4 = nn.ModuleList(
      nn.Sequential(
        Conv(x, c4, kernel_size=3),
        Conv(c4, c4, kernel_size=3),
        nn.Conv2d(c4, self.ne, kernel_size=1)
      ) for x in ch
    )

    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

  def forward(self, x):
    feat1, feat2, feat3 = self.backbone.forward(x)
    
    P5_upsample = self.upsample(feat3)
    P4          = torch.cat([P5_upsample, feat2], 1)
    P4          = self.conv3_for_upsample1(P4)

    P4_upsample = self.upsample(P4)
    P3          = torch.cat([P4_upsample, feat1], 1)
    P3          = self.conv3_for_upsample2(P3)

    P3_downsample = self.down_sample1(P3)
    P4 = torch.cat([P3_downsample, P4], 1)
    P4 = self.conv3_for_downsample1(P4)

    P4_downsample = self.down_sample2(P4)
    P5 = torch.cat([P4_downsample, feat3], 1)
    P5 = self.conv3_for_downsample2(P5)

    shape = P3.shape
    x     = [P3, P4, P5]
    x1    = [P3, P4, P5]

    bs = x[0].shape[0]
    for i in range(self.nl):
      x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)

    if self.shape != shape:
      self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
      self.shape = shape
    box, cls = torch.cat(
      [
        xi.view(shape[0], self.no, -1)
          for xi in x
      ], dim=2
    ).split(split_size=(self.reg_max * 4, self.nc), dim=1)
    dbox = self.dfl(box)
    angle = torch.cat(
      [
        self.cv4[i](x1[i]).view(bs, self.ne, -1)
          for i in range(self.nl)
      ], dim=2
    ) 
    angle = (angle.sigmoid() - 0.25) * math.pi
    return dbox, cls, x, angle, self.anchors.to(dbox.device), self.strides.to(dbox.device)
