# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:40:10 2019

@author: sbaek
"""
import cv2
import numpy as np

def poly_to_mask(polygon, shape, antialias=0):
    """Rasterizes a polygon into a binary mask.
    Arguments:
        polygon: a `numpy` array of shape `[number_of_vertices, 2]`. xy-coordinates of vertices.
        shape: a tuple of ints. shape of the mask image.
        antialias: a positive integer. creates a smooth boundary of the mask. (optional)
    Returns:
        mask: a float `numpy` array of shape `[shape[0], shape[1]]`.
    Raises:
        TODO: Exception handling
    """
    mask = np.zeros(shape=shape)
    cv2.fillPoly(mask, [polygon], 1)
    
    if antialias > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dilated = mask
        for j in range(antialias):
            dilated = cv2.dilate(dilated, kernel, iterations=1)
            mask += dilated
        mask /= np.max(mask)
    
    return mask