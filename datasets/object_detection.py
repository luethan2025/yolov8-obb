import random

import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2

from .utils import random_prob, xyxyxyxy2xywhr, flatten_bboxes

class YoloObjectDetectionDataset(Dataset):
  def __init__(
    self,
    annotations,
    input_shape,
    num_classes,
    epoch_length,
    mosaic,
    mixup,
    mosaic_prob,
    mixup_prob,
    image_set,
    augument_prob=0.7
  ):
    self.annotations   = annotations
    self.input_shape   = input_shape
    self.num_classes   = num_classes
    self.epoch_length  = epoch_length
    self.mosaic        = mosaic
    self.mosaic_prob   = mosaic_prob
    self.mixup         = mixup
    self.mixup_prob    = mixup_prob
    self.image_set     = image_set
    self.augument_prob = augument_prob

    self.epoch = 1
    self.length = len(self.annotations)

    self.bbox_attrs = 6 + num_classes

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    idx = idx % self.length
    if self.mosaic and random_prob() < self.mosaic_prob and \
      self.epoch < self.epoch_length * self.augument_prob:
      lines = random.sample(self.annotations, 3)
      lines.append(self.annotations[idx])
      random.shuffle(lines)
      image, box = self._get_random_data_mosiac(lines, self.input_shape)

      if self.mixup and random_prob() < self.mixup_prob:
        lines           = random.sample(self.annotations, 1)
        image_2, box_2  = self._get_random_data(
          lines[0], 
          self.input_shape,
          image_set=self.image_set
        )
        image, box = self._get_random_data_mixup(
          image,
          box,
          image_2,
          box_2
        )
    else:
      image, box = self._get_random_data(
        self.annotations[idx],
        self.input_shape,
        image_set=self.image_set
      )

    image = np.transpose(np.array(image, dtype=np.float32) / 255, (2, 0, 1))
    box   = np.array(box, dtype=np.float32)

    nL          = len(box)
    labels_out  = np.zeros((nL, 7))
    if nL:
      box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] / self.input_shape[1]
      box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] / self.input_shape[0]
      rbox = xyxyxyxy2xywhr(box[:, :-1])
      labels_out[:, 1] = box[:, -1]
      labels_out[:, 2:] = rbox[:, :]
    return image, labels_out
  
  def collate_fn(self, batch):
    images  = []
    bboxes  = []
    for idx, (img, box) in enumerate(batch):
      images.append(img)
      box[:, 0] = idx
      bboxes.append(box)
          
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes

  def _get_random_data(
    self,
    annotations,
    input_shape,
    hue=0.1,
    sat=0.7,
    val=0.4,
    image_set='train'
  ):
    annotation_content = annotations.split()
    image = Image.open(annotation_content[0])
    image = image.convert('RGB')

    iw, ih = image.size
    h, w = input_shape
    box = np.array(
      [
        np.array(list(map(int,box.split(','))))
          for box in annotation_content[1:]
      ]
    )
    if image_set == 'train':
      scale = min(w / iw, h / ih)
      nw = int(iw * scale)
      nh = int(ih * scale)
      dx = (w - nw) // 2
      dy = (h - nh) // 2

      image = image.resize((nw, nh), Image.BICUBIC)
      new_image = Image.new('RGB', (w, h), (128, 128, 128))
      new_image.paste(image, (dx, dy))
      image_data = np.array(new_image, np.float32)

      if len(box)>0:
        np.random.shuffle(box)
        box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
        box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy
      return image_data, box
    
    scale = random_prob(a=0.25, b=1)
    nh = int(scale * h)
    nw = int(scale * w)
    image = image.resize((nw, nh), Image.BICUBIC)

    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    flip = random_prob() < 0.5
    if flip:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image_data = np.array(image, np.uint8)
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    if len(box)>0:
      np.random.shuffle(box)
      box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
      box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy
      if flip:
        box[:, [0, 2, 4, 6]] = w - box[:, [0, 2, 4, 6]]
    return image_data, box

  def _get_random_data_mosiac(
    self,
    annotations,
    input_shape,
    hue=0.1,
    sat=0.7,
    val=0.4
  ):
    h, w = input_shape
    min_offset = 0.5

    image_datas = [] 
    box_datas   = []
    idx          = 0

    for annotation in annotations:
      annotation_content = annotation.split()
      image = Image.open(annotation_content[0])
      image = image.convert('RGB')

      iw, ih = image.size
      box = np.array(
        [
          np.array(list(map(int,box.split(','))))
            for box in annotation_content[1:]
        ]
      )

      flip = random_prob() < 0.5
      if flip and len(box) > 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        box[:, [0, 2, 4, 6]] = iw - box[:, [0, 2, 4, 6]]

      scale = 0.5
      nh = int(scale * h)
      nw = int(scale * w)
      image = image.resize((nw, nh), Image.BICUBIC)

      if idx == 0:
        dx = int(w * min_offset) - nw
        dy = int(h * min_offset) - nh
      elif idx == 1:
        dx = int(w * min_offset) - nw
        dy = int(h * min_offset)
      elif idx == 2:
        dx = int(w * min_offset)
        dy = int(h * min_offset)
      elif idx == 3:
        dx = int(w * min_offset)
        dy = int(h * min_offset) - nh

      new_image = Image.new('RGB', (w, h), (128, 128, 128))
      new_image.paste(image, (dx, dy))
      image_data = np.array(new_image)

      idx = idx + 1
      box_data = []
      if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
        box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy
        box_data = np.zeros((len(box), 9))
        box_data[:len(box)] = box
        
      image_datas.append(image_data)
      box_datas.append(box_data)

    cutx = int(w * min_offset)
    cuty = int(h * min_offset)

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    new_image = np.array(new_image, np.uint8)
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

    hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
    dtype           = new_image.dtype
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    new_image = cv2.merge(
      (
        cv2.LUT(hue, lut_hue),
        cv2.LUT(sat, lut_sat),
        cv2.LUT(val, lut_val)
      )
    )
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
    new_boxes = flatten_bboxes(box_datas)
    return new_image, new_boxes

  def _get_random_data_mixup(self, image_1, box_1, image_2, box_2):
    new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    if len(box_1) == 0:
        new_boxes = box_2
    elif len(box_2) == 0:
        new_boxes = box_1
    else:
        new_boxes = np.concatenate([box_1, box_2], axis=0)
    return new_image, new_boxes
