import argparse
import os
import os.path

import utils.plot as plt

def get_argparser():
  parser = argparse.ArgumentParser()

  # Batch Visualization Options
  parser.add_argument('--filename', type=str, default=None)
  parser.add_argument('--rows', type=int, default=None)
  parser.add_argument('--cols', type=int, default=None)
  
  return parser

def main():
  args = get_argparser().parse_args()
  with open(args.filename, 'r') as f:
    annotations = f.readlines()
  annotations = [ann.strip() for ann in annotations]
  assert(args.rows * args.cols == len(annotations))
  images = []
  bboxes = []

  for image_annotation in annotations:
    image_box = []
    sub_annotations = image_annotation.split()
    images.append(sub_annotations[0])
    for a in sub_annotations[1:]:
      x1, y1, x2, y2, x3, y3, x4, y4 = a.split(',')[:-1]
      image_box.append([x1, y1, x2, y2, x3, y3, x4, y4])
    bboxes.append(image_box)
  plt.plot_bbox(images, bboxes, args.rows, args.cols)

if __name__ == "__main__":
  main()
