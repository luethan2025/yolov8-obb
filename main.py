import argparse
from collections import OrderedDict
import random
import os
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from utils import mkdir, tuple_argument
from datasets import YoloObjectDetectionDataset
from network import Yolo
from utils.loss import OBBLoss
from utils.ema import EMA
import utils.plot as plt

UNIFIED_LOSS_PLOT_TITLE = "Unified Loss"
BBOX_LOSS_PLOT_TITLE = "BBox Loss"
CLASSIFICATION_LOSS_PLOT_TITLE = "Classification Loss"
ANGLE_LOSS_PLOT_TITLE = "Angle Loss"
LEARNING_RATE_PLOT_TITLE = "Learning Rate"

def get_argparser():
  parser = argparse.ArgumentParser()

  # Dataset Options
  parser.add_argument('--train_annotations', type=str, default='./train_annotations.txt')
  parser.add_argument('--val_annotations', type=str, default='./val_annotations.txt')
  parser.add_argument('--train_image_size', type=tuple_argument, default='(416, 416)')
  parser.add_argument('--val_image_size', type=tuple_argument, default='(416, 416)')
  parser.add_argument('--num_classes', type=int, default=5)

  # Yolo Options
  parser.add_argument('--yolo_size', type=str, default='m', choices=['n', 's', 'm','l', 'x'])

  # Train Options
  parser.add_argument('--num_epochs', type=int, default=1000)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--lr_policy', type=str, default='cosine_annealing', choices=['cosine_annealing'])
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--train_batch_size', type=int, default=64)
  parser.add_argument('--val_batch_size', type=int, default=64)
  parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
  parser.add_argument('--random_seed', type=int, default=0)

  parser.add_argument('--mosaic', action='store_true', default=True)
  parser.add_argument('--mixup', action='store_true', default=True)
  parser.add_argument('--mosaic_prob', type=float, default=0.5)
  parser.add_argument('--mixup_prob', type=float, default=0.5)
  parser.add_argument('--augment_prob', type=float, default=0.7)

  parser.add_argument('--bbox_loss_weight', type=float, default=7.5)
  parser.add_argument('--cls_loss_weight', type=float, default=0.5)
  parser.add_argument('--angle_loss_weight', type=float, default=1.5)

  # Plotting Options
  parser.add_argument('--plot', action='store_true', default=False)

  # Resuming Options
  parser.add_argument('--resume_training', action='store_true', default=False)
  parser.add_argument('--ckpt', type=str, default=None)

  return parser

def get_train_val_dataset(args):
  train_image_size = args.train_image_size
  print('\nTrain image size: %s' % str(train_image_size))
  with open(args.train_annotations, encoding='utf-8') as f:
    train_annotations = f.readlines()
  train_dst = YoloObjectDetectionDataset(
    train_annotations,
    train_image_size,
    args.num_classes,
    args.num_epochs,
    mosaic=args.mosaic,
    mixup=args.mixup,
    mosaic_prob=args.mosaic_prob,
    mixup_prob=args.mixup_prob,
    image_set='train',
    augument_prob=args.augment_prob,
  )

  val_image_size = args.val_image_size
  print('Val image size: %s' % str(args.val_image_size))
  with open(args.val_annotations, encoding='utf-8') as f:
    val_annotations = f.readlines()
  val_dst = YoloObjectDetectionDataset(
    val_annotations,
    val_image_size,
    args.num_classes,
    args.num_epochs,
    mosaic=False,
    mixup=False,
    mosaic_prob=0,
    mixup_prob=0,
    image_set='val',
    augument_prob=0,
  )
  return train_dst, val_dst

def main():
  args = get_argparser().parse_args()
  mkdir(args.ckpt_dir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device: %s' % device)

  torch.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  train_dst, val_dst = get_train_val_dataset(args)
  train_loader = data.DataLoader(
    train_dst,
    batch_size=args.train_batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    collate_fn=train_dst.collate_fn
  )
  val_loader = data.DataLoader(
    val_dst,
    batch_size=args.val_batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    collate_fn=val_dst.collate_fn
  )

  print('\nInitializing a YOLOv8%s model' % args.yolo_size)
  model = Yolo(args.num_classes, args.yolo_size)
  criterion = OBBLoss(model)
  optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epochs//10)

  ema = EMA(model)

  if args.ckpt is not None and os.path.isfile(args.ckpt):
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model = nn.DataParallel(model)
    model.to(device)
    if args.resume_training:
      optimizer.load_state_dict(checkpoint['optimizer_state'])
      scheduler.load_state_dict(checkpoint['scheduler_state'])
      ema.ema_model.load_state_dict(checkpoint['ema_model_state'])
      best_unified_loss = checkpoint['best_unified_loss']
      print('Training state restored from %s' % args.ckpt)
    print('Model restored from %s' % args.ckpt)
    del checkpoint
  else:
    print('[!] Retrain')
    best_unified_loss = float('inf')
    best_bbox_loss = float('inf')
    best_cls_loss = float('inf')
    best_angle_loss = float('inf')
    model = nn.DataParallel(model)
    model.to(device)

  def save_ckpt(path):
    torch.save({
      'model_state': ema.base_model.state_dict(),
      'ema_model_state': ema.ema_model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'scheduler_state': scheduler.state_dict(),
      'best_unified_loss': best_unified_loss,
      'best_bbox_loss': best_bbox_loss,
      'best_cls_loss': best_cls_loss, 
      'best_angle_loss': best_angle_loss
    }, path)
    print('Model saved as %s' % path)

  if args.plot:
    plots = plt.create_plots(
      2,
      3, 
      titles=[
        UNIFIED_LOSS_PLOT_TITLE,
        BBOX_LOSS_PLOT_TITLE,
        CLASSIFICATION_LOSS_PLOT_TITLE,
        ANGLE_LOSS_PLOT_TITLE,
        LEARNING_RATE_PLOT_TITLE
      ]
    )

  for epoch in range(args.num_epochs):
    train_epoch_unified_loss = 0
    train_epoch_bbox_loss = 0
    train_epoch_cls_loss = 0
    train_epoch_angle_loss = 0
    curr_lr = float(optimizer.param_groups[0]['lr'])

    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    model.train()
    for iter, batch in enumerate(train_loader):
      images, bboxes = batch
      optimizer.zero_grad()
      outputs = model(images)
      train_loss = criterion(outputs, bboxes)

      train_loss[0] *= args.bbox_loss_weight
      train_loss[1] *= args.cls_loss_weight
      train_loss[2] *= args.angle_loss_weight

      train_bbox_loss = train_loss[0]
      train_cls_loss = train_loss[1]
      train_angle_loss = train_loss[2]

      train_epoch_bbox_loss += train_bbox_loss.item()
      train_epoch_cls_loss += train_cls_loss.item()
      train_epoch_angle_loss += train_angle_loss.item()
      
      train_loss = train_loss.sum()
      train_loss.backward()
      optimizer.step()
      train_epoch_unified_loss += train_loss.item()

      ema.update(model)

      batch_bar.set_postfix(OrderedDict
        ([
          ("loss", "{:.04f}".format(float(train_epoch_unified_loss / (iter + 1)))),
          ("bbox_loss", "{:.04f}".format(float(train_epoch_bbox_loss / (iter + 1)))),
          ("cls_loss", "{:.04f}".format(float(train_epoch_cls_loss / (iter + 1)))),
          ("angle_loss", "{:.04f}".format(float(train_epoch_angle_loss / (iter + 1))))
        ])
      )
      batch_bar.update()

    print("Validating...")
    val_epoch_unified_loss = 0
    val_epoch_bbox_loss = 0
    val_epoch_cls_loss = 0
    val_epoch_angle_loss = 0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

    model = ema.ema_model
    model.eval()
    with torch.no_grad():
      for iter, batch in enumerate(val_loader):
        images, bboxes = batch
        outputs = model(images)
        val_loss = criterion(outputs, bboxes)

        val_loss[0] *= args.bbox_loss_weight
        val_loss[1] *= args.cls_loss_weight
        val_loss[2] *= args.angle_loss_weight

        val_bbox_loss = val_loss[0]
        val_cls_loss = val_loss[1]
        val_angle_loss = val_loss[2]

        val_epoch_bbox_loss += val_bbox_loss.item()
        val_epoch_cls_loss += val_cls_loss.item()
        val_epoch_angle_loss += val_angle_loss.item()

        val_loss = val_loss.sum()
        val_epoch_unified_loss += val_loss.item()

        batch_bar.set_postfix(OrderedDict
          ([
            ("loss", "{:.04f}".format(float(val_epoch_unified_loss / (iter + 1)))),
            ("bbox_loss", "{:.04f}".format(float(val_epoch_bbox_loss / (iter + 1)))),
            ("cls_loss", "{:.04f}".format(float(val_epoch_cls_loss / (iter + 1)))),
            ("angle_loss", "{:.04f}".format(float(val_epoch_angle_loss / (iter + 1))))
          ])
        )
        batch_bar.update()

    scheduler.step()
    print('Epoch %d/%d' % (epoch, args.num_epochs))
    print('Train Unified Loss: %.4f Train Regression Loss: %.4f Train Classification Loss: %.4f Train Angle Loss: %.4f' % (train_epoch_unified_loss/len(train_loader), train_epoch_bbox_loss/len(train_loader), train_epoch_cls_loss/len(train_loader), train_epoch_angle_loss/len(train_loader)))
    print('Val Unified Loss: %.4f Val Regression Loss: %.4f Val Classification Loss: %.4f Val Angle Loss: %.4f' % (val_epoch_unified_loss/len(val_loader), val_epoch_bbox_loss/len(val_loader), val_epoch_cls_loss/len(val_loader), val_epoch_angle_loss/len(val_loader)))

    save_ckpt('checkpoints/last_epoch_yolov8%s.pth' % (args.yolo_size))
    val_unified_loss = val_epoch_unified_loss / len(val_loader)
    if val_unified_loss < best_unified_loss:
      best_unified_loss = val_unified_loss
      save_ckpt('checkpoints/best_unified_yolov8%s.pth' % (args.yolo_size))
  
    val_bbox_loss = val_epoch_bbox_loss / len(val_loader)
    if val_bbox_loss < best_bbox_loss:
      best_bbox_loss = val_bbox_loss
      save_ckpt('checkpoints/best_bbox_yolov8%s.pth' % (args.yolo_size))

    val_cls_loss = val_epoch_cls_loss / len(val_loader)
    if val_cls_loss < best_cls_loss:
      best_cls_loss = val_cls_loss
      save_ckpt('checkpoints/best_cls_yolov8%s.pth' % (args.yolo_size))

    val_angle_loss = val_epoch_angle_loss / len(val_loader)
    if val_angle_loss < best_angle_loss:
      best_angle_loss = val_angle_loss
      save_ckpt('checkpoints/best_angle_yolov8%s.pth' % (args.yolo_size))

    model = ema.base_model

    if args.plot:
      print("Updating plots...")
      new_data = {
        UNIFIED_LOSS_PLOT_TITLE: ([epoch], [val_unified_loss]),
        BBOX_LOSS_PLOT_TITLE: ([epoch], [val_bbox_loss]),
        CLASSIFICATION_LOSS_PLOT_TITLE: ([epoch], [val_cls_loss]),
        ANGLE_LOSS_PLOT_TITLE: ([epoch], [val_angle_loss]),
        LEARNING_RATE_PLOT_TITLE: ([epoch], [curr_lr])
      }
      plt.update_plots(plots, new_data)

if __name__ == '__main__':
  main()
