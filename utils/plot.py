import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import cv2
from matplotlib.patches import Polygon

def create_plots(rows, cols, titles, figsize=(20,7)):
  plots = {}
  _, axarr = plt.subplots(rows, cols, figsize=figsize)
  axarr = np.array(axarr).reshape(-1)
  for idx, ax in enumerate(axarr):
    ax.axis('on')
    ax.set_autoscale_on(True)
    if titles:
      try:
        ax.set_title(titles[idx])
        plots[titles[idx]] = ax
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
      except:
        ax.axis('off')
        break
  plt.tight_layout()
  plt.draw()
  plt.pause(0.1)
  return plots

def update_plots(plots, new_data):
  for title, value in new_data.items():
    if title in plots:
      ax = plots[title]
      ax.set_autoscale_on(True)
      lines = ax.get_lines()
      x, y = value
      if len(lines) == 0:
        ax.plot(x, y)
      else:
        lines = ax.get_lines()
        ax.clear()
        for l in lines:
          xs = l.get_xdata()
          ys = l.get_ydata()
          xs = np.append(xs, x)
          ys = np.append(ys, y)
          ax.plot(xs, ys)
      ax.set_title(title)
      ax.set_xlim(left=0)
      ax.xaxis.set_major_locator(MaxNLocator(integer=True))
      ax.set_ylim(bottom=0)
      ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def plot_bbox(images, bboxes, rows, cols, figsize=(20,7)):
  _, axarr = plt.subplots(rows, cols, figsize=figsize)
  axarr = np.array(axarr).reshape(-1) 
  for i, ax in enumerate(axarr):
    ax.imshow(cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB))
    ax.axis('off')
    for b in bboxes[i]:
      x1, y1, x2, y2, x3, y3, x4, y4 = b
      points = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
      polygon = Polygon(points, linewidth=2, edgecolor='red', facecolor='none')
      ax.add_patch(polygon)

  plt.tight_layout()
  plt.show()
