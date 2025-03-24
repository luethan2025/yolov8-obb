import math
import copy

import torch
import torch.nn as nn

def copy_attr(a, b, include=(), exclude=()):
  for k, v in b.__dict__.items():
    if (len(include) and k not in include) or k.startswith("_") or k in exclude:
      continue
    else:
      setattr(a, k, v)

def is_parallel(model):
  return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
  return model.module if is_parallel(model) else model

class EMA:
  def __init__(self, model, decay=0.9999, tau=2000, updates=0):
    self.base_model = model
    self.ema_model = copy.deepcopy(de_parallel(model)).eval()
    self.updates = updates
    self.decay = lambda x: decay * (1 - math.exp(-x / tau))
    for p in self.ema_model.parameters():
      p.requires_grad_(False)
    self.enabled = True

  def update(self, model):
    if self.enabled:
      self.updates += 1
      d = self.decay(self.updates)

      msd = de_parallel(model).state_dict()
      for k, v in self.ema_model.state_dict().items():
        if v.dtype.is_floating_point:
          v *= d
          v += (1 - d) * msd[k].detach()

  def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
    if self.enabled:
      copy_attr(self.ema_model, model, include, exclude)
