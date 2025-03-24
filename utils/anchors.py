import torch

def make_anchors(features, strides, grid_cell_offset=0.5):
  dtype = features[0].dtype
  device = features[0].device
  anchor_points = []
  stride_tensor = []
  for i, stride in enumerate(strides):
    _, _, h, w = features[i].shape
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
    sy, sx = torch.meshgrid(sy, sx, indexing="ij")
    anchor_points.append(
      torch.stack((sx, sy), dim=-1).view(-1, 2)
    )
    stride_tensor.append(
      torch.full(
        size=(h * w, 1),
        fill_value=stride,
        dtype=dtype,
        device=device
      )
    )
  return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)
