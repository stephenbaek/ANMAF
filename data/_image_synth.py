# Copyright (c) 2019 Visual Intelligence Laboratory. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Cell image synthesis from background templates and annotated cells."""

from _common import _roi_paths, _image_dir, _background_dir
import os
import pathlib
import numpy as np
import cv2
from read_roi import read_roi_zip

import json

import utils.tiff
from utils.point_sampling import poisson_disc_samples
from utils.image_processing import poly_to_mask

from tqdm import tqdm
import random
import math



def _load_annotations():
    """Load all annotations from `roi_paths`.
       Annotation files are provided as ImageJ ROI files (*.zip).
       Corresponding images must be in a subdirectory named `<image_dir_name>` under the directory where the roi zip file is in.
    Arguments:
        image_dir_name: a string. subdirectory name where images to be loaded
    Returns:
        annotations: a list of dictionaries. keys include `case_name`, `rois`, `images`.
            - `case_name` is a list of strings containing the names of roi zip directories.
            - `rois` is a list of dictionaries containing roi information loaded from zip files.
            - `images` is a list of images that corresponds to the rois.
    Raises:
        TODO: Exception handling
    """
    # Read all annotation files and corresponding images
    annotations = []
    for roi_path in _roi_paths:
        # Parse paths
        image_paths = pathlib.Path(os.path.join(*[os.path.split(roi_path)[0], _image_dir]))
        image_paths = [str(path) for path in list(image_paths.glob('*.tif'))]
        case_name = os.path.split(os.path.split(roi_path)[0])[1]
        
        # Read ROI data
        rois = read_roi_zip(roi_path)
            
        # Indices in the file paths are not sorted initially.
        idx = []
        for image_path in image_paths:
            splitted_filename = (os.path.split(image_path)[1]).split('_')
            idx.append( int(splitted_filename[1]) )
        
        image_paths = sorted(zip(idx, image_paths))
        
        # Read z-projection images
        images = []
        for idx, image_path in image_paths:
            image = utils.tiff.load_tiff(image_path)
            images.append(image)
    
        annotations.append({'case_name': case_name, 'rois': rois, 'images': images})
        
    return annotations

def _crop_cells(annotations, antialias=10):
    """Scissors cell images along the boundary contour.
    Arguments:
        annotations: a list of dictionaries. keys include `case_name`, `rois`, `images`.
            - `case_name` is a list of strings containing the names of roi zip directories.
            - `rois` is a list of dictionaries containing roi information loaded from zip files.
            - `images` is a list of images that corresponds to the rois.
        antialias: a positive integer. when non-zero, it creates a smooth transition of pixel values at the boundary. (optional)
    Returns:
        cells: a list of dictionaries. keys include `image`, `mask`, `polygon`, `name`, `case_name`.
            - `image`: a `numpy` array of shape `[image_height, image_width]`. cropped cell image.
            - `mask`: a `numpy` array of the same shape. the mask on the cell area.
            - `polygon`: a `numpy` array of shape `[number_of_vertices, 2]`. the xy-coordinates of the contour polygon vertices.
            - `name`: a string. name of the cell.
            - `case_name`: a string. case identifier.
    Raises:
        TODO: Exception handling
    """
    cells = []
    for annotation in annotations:
        case_name = annotation['case_name']
        rois = annotation['rois']
        images = annotation['images']
        
        # Now visit each annotation and extract the corresponding subimage from z-proj
        for key, roi in rois.items():
            # find corresponding image to the current roi
            z = (roi['position']-1 )// 10
            
            if (z > len(images)-1):
                z = z-1
                
        
            image = images[z]
            
            # polygon coordinates
            x = roi['x']
            y = roi['y']
            
            # Compute range of pixels after antialiasing
            # One step of antialiasing adds a strip of one pixel to each side.
            xmin = np.min(x) - antialias
            xmax = np.max(x) + antialias + 1
            ymin = np.min(y) - antialias
            ymax = np.max(y) + antialias + 1
            
            # If the cell area goes beyond the image domain, discard the cell
            # TODO: What about just zero-padding or repeated pixels?
            if xmin < 0 or xmax >= image.shape[1] or ymin < 0 or ymax >= image.shape[0]:
                continue
            
            # Crop the cell area
            cell_image = image[ymin:ymax, xmin:xmax]
            
            # Convert coordinates into the new (cropped) coordinate system
            cell_contour = np.array([x-xmin, y-ymin]).T
            
            # Create a binary mask with smoothened boundary (1: cell, <1: background)
            cell_mask = poly_to_mask(cell_contour, cell_image.shape, antialias)
            
            
            cells.append({'image': cell_image, 'mask': cell_mask, 'polygon': cell_contour, 'name': key, 'case_name': case_name})
            
    return cells

def _read_random_background():
    # File paths to synthetic backgrounds
    _background_paths = [str(path) for path in list(pathlib.Path(_background_dir).glob('*.tif'))]
    background = utils.tiff.load_tiff(_background_paths[np.random.randint(0,len(_background_paths)-1)])
    background *= np.random.uniform(0.8,3)
    return background

   
def synthesize_images(n, path):
    annotations = _load_annotations()
    cells = _crop_cells(annotations, 5)
    
    ####################################################################    
    # seperate training and validation cells
    sorted(cells, key = lambda i: (i['case_name'], i['name'])) 
    random.seed(666)
    random.shuffle(cells)
    
    cutoff = 0.2
    cutoff_num = math.ceil(cutoff*len(cells))
    
    if ('train' in path):
        cells = cells[cutoff_num : ]
        #############################
        # select a subset of them for experiment
        percent = 1
        percent_idx = math.ceil(percent*len(cells))
        cells = cells[ : percent_idx]
        #print('number of cells used is: ', len(cells))
        
    elif ('val' in path):
        cells = cells[ : cutoff_num]
    else:
        pass
    
    
    ########################################################################
    
    CELL_MEAN_RADIUS_MIN = 40  # This controls the number of cells to be generated in the synthetic image 
    CELL_MEAN_RADIUS_MAX = 60  # IMPORTANT: big number = lesser cells, small number = more cells
    
    max_cell_shape = np.max([cell['image'].shape for cell in cells], axis=0)
    
        
    data = {}
    print('Generating synthetic images...')
    for i in tqdm(range(n)):
        # Pick a random background
        image = _read_random_background()
        height, width = image.shape
        
        
        cell_mean_radius = np.random.randint(CELL_MEAN_RADIUS_MIN, CELL_MEAN_RADIUS_MAX)
        p = poisson_disc_samples(width-0.7*max_cell_shape[1], height-0.7*max_cell_shape[0], r=cell_mean_radius)
        tx = p[:,0]
        ty = p[:,1]
        
        NUM_CELLS = len(tx)
        cell_ids = np.random.randint(0, len(cells), NUM_CELLS)
        
        # Put zeros on the side. Sometimes cell images go beyond the image region.
        image = np.concatenate((image, np.zeros((image.shape[0], max_cell_shape[1]))), axis=1)
        image = np.concatenate((image, np.zeros((max_cell_shape[0], image.shape[1]))), axis=0)
        
        contours = []
        for j, cid in enumerate(cell_ids):
            cell = cells[cid]
            cell_image = np.copy(cell['image'])
            cell_mask = np.copy(cell['mask'])
            cell_contour = np.copy(cell['polygon'])
            
            # Random 90 rotation
            if np.random.uniform() > 0.5:
                cell_image = cell_image.T
                cell_mask = cell_mask.T
                cell_contour = cell_contour[:,[1,0]]
                
            # Random flip left and right
            if np.random.uniform() > 0.5:
                cell_image = np.fliplr(cell_image)
                cell_mask = np.fliplr(cell_mask)
                cell_contour[:,0] = cell_image.shape[1] - cell_contour[:,0]
            
            # Random flip upside down
            if np.random.uniform() > 0.5:
                cell_image = np.flipud(cell_image)
                cell_mask = np.flipud(cell_mask)
                cell_contour[:,1] = cell_image.shape[0] - cell_contour[:,1]
                            
            # Random resize
            if np.random.uniform() > 0.5:
                sx = np.random.uniform(0.8, 1.2)
                sy = np.random.uniform(0.8, 1.2)
                cell_image = cv2.resize(cell_image, None, fx=sx, fy=sy)
                cell_mask = cv2.resize(cell_mask, None, fx=sx, fy=sy)
                cell_contour[:, 0] = np.round(cell_contour[:, 0]*sx).astype('int32')
                cell_contour[:, 1] = np.round(cell_contour[:, 1]*sy).astype('int32')
                
            # Random contrast
            if np.random.uniform() > 0.5:
                cell_image *= np.random.uniform(0.95, 1.3)
                    
            subimage = image[ty[j]:ty[j]+cell_image.shape[0], tx[j]:tx[j]+cell_image.shape[1]]
            intensity_background = np.mean(subimage)
            intensity_cellimage = np.mean(cell_image)
            
            contrast_factor = 1.0
            if intensity_background*1.5 > intensity_cellimage:
                contrast_factor = 1.5*intensity_background/intensity_cellimage
            image[ty[j]:ty[j]+cell_image.shape[0], tx[j]:tx[j]+cell_image.shape[1]] = np.maximum(cell_image*cell_mask*contrast_factor, subimage)
            
            cell_contour += [tx[j], ty[j]]
            cell_contour[:,0] = np.clip(cell_contour[:,0], 0, width-1)
            cell_contour[:,1] = np.clip(cell_contour[:,1], 0, height-1)
            contours.append(cell_contour)
        
        image = image[0:height, 0:width]
        image_name = ('%08d.tif')%i
        
        image_path = os.path.join(*[path, image_name])
        utils.tiff.save_tiff(image, image_path)
        filesize = os.path.getsize(image_path)
        
        data_key = 'synthetic_' + image_name + ('%d')%filesize
        data[data_key] = {
                'fileref': "",
                'size': filesize,
                'filename': image_name,
                'base64_img_data': "",
                'file_attributes': {},
                'regions': {}
                }
        for j, cid in enumerate(cell_ids):
            key = ('%03d_')%j + cells[cid]['case_name'] + '_' + cells[cid]['name']
            x = contours[j][:,0].tolist()
            y = contours[j][:,1].tolist()
            data[data_key]['regions'][key] = {
                            'shape_attributes':{
                                    'name': 'polygon',
                                    'all_points_x': x,
                                    'all_points_y': y
                                    },
                            'region_attributes': {}
                            }
                            
    
    # Exports VGG Image Annotator (VIA) json format (http://www.robots.ox.ac.uk/~vgg/software/via/)
    # This is the format that M-RCNN implementation expects
    # TODO: Export in imagej roi (https://imagej.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html)
    print('Exporting annotation file...')
    with open(os.path.join(*[path, 'via_region_data.json']), 'w') as file:  
        json.dump(data, file)
        
