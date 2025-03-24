import math

import torch

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
  if xywh:
    x1, y1, w1, h1 = box1.chunk(chunks=4, dim=-1)
    x2, y2, w2, h2 = box2.chunk(chunks=4, dim=-1)

    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
  else:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(chunks=4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(chunks=4, dim=-1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

  inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
    b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
  ).clamp_(0)

  union = w1 * h1 + w2 * h2 - inter + eps

  iou = inter / union
  if CIoU or DIoU or GIoU:
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    if CIoU or DIoU:
      c2 = cw.pow(2) + ch.pow(2) + eps
      rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
      ) / 4
      if CIoU:
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
          alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
      return iou - rho2 / c2
    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area
  return iou

def _get_covariance_matrix(boxes):
  gbbs = torch.cat((boxes[:, 2: 4].pow(2) / 12, boxes[:, 4:]), dim=-1)
  a, b, c = gbbs.split(1, dim=-1)
  cos = c.cos()
  sin = c.sin()
  cos2 = cos.pow(2)
  sin2 = sin.pow(2)
  return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def prob_iou(obb1, obb2, CIoU=False, eps=1e-7):
  x1, y1 = obb1[..., :2].split(1, dim=-1)
  x2, y2 = obb2[..., :2].split(1, dim=-1)
  a1, b1, c1 = _get_covariance_matrix(obb1)
  a2, b2, c2 = _get_covariance_matrix(obb2)

  t1 = (
    ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / 
      ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
  ) * 0.25
  t2 = (
    ((c1 + c2) * (x2 - x1) * (y1 - y2)) / 
      ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
  ) * 0.5
  t3 = (
    ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / 
      (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
    + eps
  ).log() * 0.5
  bd = (t1 + t2 + t3).clamp(eps, 100.0)
  hd = (1.0 - (-bd).exp() + eps).sqrt()
  iou = 1 - hd
  if CIoU:
    w1, h1 = obb1[..., 2: 4].split(split_size_or_sections=1, dim=-1)
    w2, h2 = obb2[..., 2: 4].split(split_size_or_sections=1, dim=-1)
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
      alpha = v / (v - iou + (1 + eps))
    return iou - v * alpha
  return iou
