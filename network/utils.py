import torch

def compute_padding(kernel_size, padding=None, dilation=1):
  if dilation > 1:
    if isinstance(kernel_size, int):
      kernel_size = dilation * (kernel_size - 1) + 1
    else:
      kernel_size = [dilation * (k - 1) + 1 for k in kernel_size]
  if padding is None:
    if isinstance(kernel_size, int):
      padding = kernel_size // 2
    else:
      padding = [k // 2 for k in kernel_size]
  return padding

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points = []
  stride_tensor = []
  dtype = feats[0].dtype
  device = feats[0].device
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
    sy, sx = torch.meshgrid(sy, sx, indexing="ij")
    anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
    stride_tensor.append(
      torch.full(
        size=(h * w, 1),
        fill_value=stride,
        dtype=dtype,
        device=device
      )
    )
  return torch.cat(anchor_points), torch.cat(stride_tensor)
