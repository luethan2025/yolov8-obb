import torch
import numpy as np

def bbox2dist(anchor_points, bbox, reg_max):
  x1y1, x2y2 = bbox.chunk(2, -1)
  return torch.cat(
    (anchor_points - x1y1, x2y2 - anchor_points), dim=-1
  ).clamp_(0, reg_max - 0.01)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(chunks=2, dim=dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((xy, wh), dim=dim)
  return torch.cat((x1y1, x2y2), dim=dim)

def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
  lt, rb = pred_dist.split(2, dim=dim)
  cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
  xf, yf = ((rb - lt) / 2).split(1, dim=dim)
  x, y = xf * cos - yf * sin, xf * sin + yf * cos
  xy = torch.cat([x, y], dim=dim) + anchor_points
  return torch.cat([xy, lt + rb], dim=dim)

def xywh2xyxy(x):
  y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
  xy = x[..., :2]
  wh = x[..., 2:] / 2
  y[..., :2] = xy - wh
  y[..., 2:] = xy + wh
  return y

def xywhr2xyxyxyxy(x):
  cos, sin, cat, stack = (
    (torch.cos, torch.sin, torch.cat, torch.stack)
    if isinstance(x, torch.Tensor)
    else (np.cos, np.sin, np.concatenate, np.stack)
  )

  ctr = x[..., :2]
  w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
  cos_value, sin_value = cos(angle), sin(angle)
  vec1 = [ w / 2 * cos_value, w / 2 * sin_value]
  vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
  vec1 = cat(vec1, -1)
  vec2 = cat(vec2, -1)
  pt1 = ctr + vec1 + vec2
  pt2 = ctr + vec1 - vec2
  pt3 = ctr - vec1 - vec2
  pt4 = ctr - vec1 + vec2
  return stack([pt1, pt2, pt3, pt4], -2)
