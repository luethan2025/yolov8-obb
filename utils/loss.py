import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou, prob_iou
from .bbox import bbox2dist, dist2bbox, dist2rbox, xywh2xyxy
from .anchors import make_anchors
from .assigner import TaskAlignedAssigner, RotatedTaskAlignedAssigner

class BboxLoss(nn.Module):
  def __init__(self, reg_max, use_dfl=False):
    super().__init__()
    self.reg_max = reg_max
    self.use_dfl = use_dfl

  def forward(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask
  ):
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    if self.use_dfl:
      target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
      loss_dfl = self._df_loss(
        pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]
      ) * weight
      loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
      loss_dfl = torch.tensor(0.0).to(pred_dist.device)
    return loss_iou, loss_dfl
  
  @staticmethod
  def _df_loss(pred_dist, target):
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1 - wl
    return (
      F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl \
        + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
    ).mean(dim=-1, keepdim=True)

class RotatedBboxLoss(BboxLoss):
  def __init__(self, reg_max, use_dfl=False):
    super().__init__(reg_max, use_dfl)

  def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = prob_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    if self.use_dfl:
      target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
      loss_dfl = self._df_loss(
        pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]
      ) * weight
      loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
      loss_dfl = torch.tensor(0.0).to(pred_dist.device)
    return loss_iou, loss_dfl

class DetectionLoss:
  def __init__(self, model, tal_topk=10):
    device = next(model.parameters()).device
    self.bce = nn.BCEWithLogitsLoss(reduction="none")
    self.stride = model.stride
    self.nc = model.nc
    self.no = model.nc + model.reg_max * 4
    self.reg_max = model.reg_max
    self.device = device

    self.use_dfl = model.reg_max > 1

    self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
    self.bbox_loss = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl).to(device)
    self.proj = torch.arange(model.reg_max, dtype=torch.float, device=device)

  def preprocess(self, targets, batch_size, scale_tensor):
    if targets.shape[0] == 0:
      out = torch.zeros(batch_size, 0, 5, device=self.device)
    else:
      i = targets[:, 0]
      _, counts = i.unique(return_counts=True)
      counts = counts.to(dtype=torch.int32)
      out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
      for j in range(batch_size):
        matches = i == j
        n = matches.sum()
        if n:
          out[j, :n] = targets[matches, 1:]
      out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out

  def bbox_decode(self, anchor_points, pred_dist):
    if self.use_dfl:
      b, a, c = pred_dist.shape
      pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=False)

  def __call__(self, preds, batch):
    loss = torch.zeros(3, device=self.device)
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat(
      [
        xi.view(feats[0].shape[0], self.no, -1)
          for xi in feats
      ], dim=2
    ).split(
      (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
      pred_scores.detach().sigmoid(),
      (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
      anchor_points * stride_tensor,
      gt_labels,
      gt_bboxes,
      mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

    if fg_mask.sum():
      target_bboxes /= stride_tensor
      loss[0], loss[2] = self.bbox_loss(
        pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
      )

    loss[0] *= 7.5
    loss[1] *= 0.5
    loss[2] *= 1.5

    return loss.sum() * batch_size, loss.detach()

class OBBLoss(DetectionLoss):
  def __init__(self, model):
    super().__init__(model)
    self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
    self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

  def preprocess(self, targets, batch_size, scale_tensor):
    if targets.shape[0] == 0:
      out = torch.zeros(batch_size, 0, 6, device=targets.device)
    else:
      i = targets[:, 0]
      _, counts = i.unique(return_counts=True)
      counts = counts.to(dtype=torch.int32)
      out = torch.zeros(batch_size, counts.max(), 6, device=targets.device)
      for j in range(batch_size):
        matches = i == j
        n = matches.sum()
        if n:
          bboxes = targets[matches, 2:]
          bboxes[..., :4].mul_(scale_tensor)
          out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
    return out

  def __call__(self, preds, batch):
    device = preds[1].device
    loss = torch.zeros(3, device=device)
    feats = preds[2] if isinstance(preds, tuple) else preds
    pred_angle = preds[3]
    batch_size = pred_angle.shape[0]
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
      (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    pred_angle = pred_angle.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    targets = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
    rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
    targets = targets[(rw >= 2) & (rh >= 2)]
    targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 5), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

    bboxes_for_assigner = pred_bboxes.clone().detach()
    bboxes_for_assigner[..., :4] *= stride_tensor
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
      pred_scores.detach().sigmoid(),
      bboxes_for_assigner.type(gt_bboxes.dtype),
      anchor_points * stride_tensor,
      gt_labels,
      gt_bboxes,
      mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

    if fg_mask.sum():
      target_bboxes[..., :4] /= stride_tensor
      loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
    else:
      loss[0] += (pred_angle * 0).sum()
    return loss

  def bbox_decode(self, anchor_points, pred_dist, pred_angle):
    if self.use_dfl:
      b, a, c = pred_dist.shape
      pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
    return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
