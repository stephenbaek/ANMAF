# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:59:19 2019

@author: sbaek
"""


import tifffile
import numpy as np


def load_tiff(path):
    img = tifffile.imread(path)
    img = img.astype('float32')
    return img

def save_tiff(img, path):
    img = np.clip(img, a_min=0, a_max=None).astype('uint16')
    tifffile.imsave(path, img)