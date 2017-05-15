import numpy as np
from scipy import misc
import glob
import os
from libs import gif

# Start to scan images on by one
imgs = []
for image_path in glob.glob("./result_shapenet_clvae/reconstruction_*.png"):
    image = misc.imread(image_path)
    imgs.append(image)
gif.build_gif(imgs, saveto='ae.gif', cmap='grey')
