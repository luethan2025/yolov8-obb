import argparse
import os
import os.path
import re

from PIL import Image

from utils import tuple_argument, listdir, write

def get_argparser():
  parser = argparse.ArgumentParser()

  # Dataset Options
  parser.add_argument('--images_dir', type=str, default=None)
  parser.add_argument('--labels_dir', type=str, default=None)
  parser.add_argument('--image_size', type=tuple_argument, default='(416, 416)')
  parser.add_argument('--dst', type=str, default=None)

  return parser

def main():
  args = get_argparser().parse_args()
  BASE_TEMPLATE = "%s"
  OBB_TEMPLATE = "%d,%d,%d,%d,%d,%d,%d,%d,%s"

  images = listdir(args.images_dir)
  labels = listdir(args.labels_dir)
  labels = [
    re.sub(rf'{os.path.abspath(args.labels_dir)}/', '', l)[:-4]
      for l in labels
  ]

  class_labels = set()
  annotations = []
  image_counter = 0
  for img in images:
    base_name = re.sub(rf'{os.path.abspath(args.images_dir)}/', '', img)[:-4]
    if base_name in labels:
      img_file = os.path.join(os.path.abspath(args.images_dir), base_name + '.jpg')
      with Image.open(img_file) as imgf:
        iw, ih = imgf.size
        if (iw, ih) != args.image_size:
          continue
      image_counter += 1
      label_file = os.path.join(os.path.abspath(args.labels_dir), base_name + '.txt')

      sub_annotation = [BASE_TEMPLATE % (img_file)]
      with open(label_file, 'r') as lf:
        lines = lf.readlines()
        for l in lines:
          l = l.strip()
          cl, x1, y1, x2, y2, x3, y3, x4, y4 = l.split()
          x1, x2, x3, x4 = map(float, [x1, x2, x3, x4])
          x1, x2, x3, x4 = map(lambda x: x * iw, [x1, x2, x3, x4])
          x1, x2, x3, x4 = map(int, [x1, x2, x3, x4])

          y1, y2, y3, y4 = map(float, [y1, y2, y3, y4])
          y1, y2, y3, y4 = map(lambda x: x * ih, [y1, y2, y3, y4])
          y1, y2, y3, y4 = map(int, [y1, y2, y3, y4])
          obb = OBB_TEMPLATE % (x1, y1, x2, y2, x3, y3, x4, y4, cl)
          sub_annotation.append(obb)
          class_labels.add(cl)
      annotations.append(' '.join(sub_annotation))
  class_labels = sorted(list(class_labels))
  print("Seen the %d labels (%s) in %d images from %s" % (len(class_labels), ', '.join(class_labels), image_counter, args.images_dir))
  write(args.dst, '\n'.join(annotations))

if __name__ == '__main__':
  main()
