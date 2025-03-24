import torch
import numpy as np
import cv2

def random_prob(a=0, b=1):
  return np.random.rand() * (b - a) + a

def xyxyxyxy2xywhr(x):
  points = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
  points = points.reshape(len(x), -1, 2)
  boxes = []
  for p in points:
    (cx, cy), (w, h), angle = cv2.minAreaRect(p)
    boxes.append([cx, cy, w, h, angle / 180 * np.pi])
  return torch.tensor(boxes, device=x.device, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.asarray(boxes)

def flatten_bboxes(box_data):
  boxes = []
  for i in range(len(box_data)):
    for box in box_data[i]:
      boxes.append(box)
  return boxes
