#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import glob
import math
import numpy as np
import tensorflow as tf

import skimage
import cv2
import pandas as pd
import shutil

import mrcnn.model as modellib
from mrcnn.config import Config
from utils.tiff import load_tiff
import pathlib
from utils.roiexport import roiexport
from config import RDSS_ROOT



# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = pathlib.Path(os.path.join(*[ROOT_DIR, 'logs']))
print(MODEL_DIR)




class Config(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "soma"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + soma

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
config = Config()



# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0" 

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
# weights_path = model.find_last()
weights_path = 'anmaf.h5'


# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)



def shoelace(coords):
    corners = np.array([coords[0], coords[1]]).transpose()
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def find_contours(amask):
    temp = amask.astype('uint8')
    contours, _= cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def find_coordinates(aContour):
    temp = aContour.reshape(aContour.shape[0],2)
    x_coord = temp[:,0]
    y_coord = temp[:,1]
    return x_coord, y_coord

def centroid(arr1, arr2):
    length = arr1.shape[0]
    w = np.max(arr1) - np.min(arr1)
    h = np.max(arr2) - np.min(arr2)
    sum_x = np.sum(arr1)
    sum_y = np.sum(arr2)
    return sum_x/length, sum_y/length, w, h
    
    

def detection(data_dir, alist, result_dir, microns):
    
    t_res = microns/512   # microns per 512 pixels (transversal)
    
    a_res = t_res**2     # 1 pixel occupies t_res^2 square-microns
    z_res = 2            # 2 microns per pixel along z direction
    v_res = a_res*z_res  # 1 voxel occupies a_res*z_res cubic-microns
        
    for case in alist:
        print('###############################', case, '##################################')
        # STEP 1. CELL DETECTION
        # The code below detects bounding boxes and masks for different cells

        path = data_dir+'/'+case+'/Z-Projection/'
            
        roi_stack = []
        mask_stack = []
        tiffs = os.listdir(path)
        tiffs  = [x for x in tiffs if '.tif' in x]
        tiffs.sort(key=lambda x:int(x.split('_')[1]))
        print(tiffs)
        for tiff in tiffs:
            print(tiff)
            tiffpath = path + tiff
            img = load_tiff(tiffpath)
            img = img/np.max(img)    
            img = img*255
            temp = skimage.color.gray2rgb(img)
            result = model.detect([temp], verbose = 1)
            rois = result[0]['rois']
            print(rois.shape)
            masks = result[0]['masks']
            print(masks.shape)
            

            roi_stack.append(rois)
            mask_stack.append(masks)
                

        print('Detection is done')

        

        # STEP 2. LABLEING CELLS ON SLICES

        label = 0
        label_stack = []
        slices_num = []
        for i in range(len(tiffs)):   # for each slice
            labels = []
            for a in range(mask_stack[i].shape[2]):  # visit all the masks in the slice 
                # register everything into the labels array
                labels.append(label)
                label += 1
                slices_num.append(tiffs[i][:-4])

            label_stack.append(labels)
            print(labels)


        # Generate detected results
        result_path = result_dir + '/' + case +  '/Z_Projection_Overlayed'
            
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for i, tiff in enumerate(tiffs):
            tiffpath = path + tiff
            img = load_tiff(tiffpath)
            img = img/np.max(img)
            img = img*255

            img = img.astype('uint8')
            img = skimage.color.gray2rgb(img)  # pretend it's a color image

            masked = img.copy()

            rois = roi_stack[i]
            masks = mask_stack[i]
            for j, (mask, roi) in enumerate(zip(masks.transpose(2,0,1), rois)):
                
                contours = find_contours(mask)
                if (len(contours)>1):
                    area_lst = []
                    for acontour in contours:
                        area_lst.append(shoelace(find_coordinates(acontour)))

                    max_idx = area_lst.index(max(area_lst))
                    valid_contour = contours[max_idx]
                    valid_area = area_lst[max_idx]
                else:
                    valid_contour = contours[0]
                    valid_area = shoelace(find_coordinates(valid_contour))

                x_coords, y_coords = find_coordinates(valid_contour)
                
                boundaries = np.zeros((512,512))
                boundaries[y_coords,x_coords] = 1

                temp = masked + np.concatenate( (np.zeros((512,512,2)), np.expand_dims(boundaries*255, axis=2)), axis=2 )
                masked = np.clip(temp, 0, 255).astype('uint8')
                
                if (roi[2] < 450):
                    textloc = (roi[1]+10, roi[2]+10)
                else:
                    textloc = (roi[1], roi[0]-5)
                masked = cv2.putText(masked, 'M%d'% label_stack[i][j], textloc, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.3, (0, 255, 255), lineType=cv2.LINE_AA)

            cv2.imwrite(result_path + '/' + tiff[:-3] + 'png', masked)
 



        data = []
        
        # Store mask for each label
        total_mask = []
        
        # Create folder to rois
        roi_path = result_path + '/RoiSet'
        if not os.path.exists(roi_path):
            os.makedirs(roi_path)
        
        for i in range(label):
            num_slice = slices_num[i]
            print(num_slice)
            stack_num = (int(num_slice.split('_')[1])//10) + 1
            print(stack_num)

            tf = np.zeros((512, 512))

            for a, labels in enumerate(label_stack):
                b = np.where(np.array(labels)==i) # b = slice number of where cell i is located
                if np.size(b):

                    mask = mask_stack[a][:,:,b]
                    amask = np.squeeze(mask, axis=(2,3))
                    print(amask.shape)

                    contours = find_contours(amask)
                    if (len(contours)>1):
                        area_lst = []
                        for acontour in contours:
                            area_lst.append(shoelace(find_coordinates(acontour)))

                        max_idx = area_lst.index(max(area_lst))
                        valid_contour = contours[max_idx]
                        valid_area = area_lst[max_idx]
                    else:
                        valid_contour = contours[0]
                        valid_area = shoelace(find_coordinates(valid_contour))
                    
                    x_coords, y_coords = find_coordinates(valid_contour)
                    z_coords = stack_num
                    cen_info = centroid(x_coords, y_coords)
                    roi_name = '%03d-%03d-%03d'%(cen_info[0], cen_info[1], z_coords)
                    roiexport(roi_path + '/' + roi_name +'.roi', list(x_coords), list(y_coords), z_coords, roi_name)
                    
                    
                    mask = np.squeeze(mask)
                    tf += mask

            total_mask.append(tf)

            data.append([num_slice, valid_area, cen_info[0], cen_info[1], valid_area*a_res])

            print('label %d: %s %f %f %f'%(i, num_slice, valid_area, cen_info[0], cen_info[1]))

        df = pd.DataFrame(data)
        df.columns = ['Slice_Index', 'Shoelace Area (px^2)', 'X (px)', 'Y (px)', 'Shoelace Area (um^2)']

        df.to_csv(result_path + '/table.csv')
        total_mask = np.array(total_mask)
        total_mask = total_mask.transpose(1,2,0)
        print(total_mask.shape)
        np.save(result_path + '/masks.npy', total_mask)
        shutil.make_archive(result_path+'/'+'RoiSet', 'zip', roi_path)
        print('Results Saved')







data_folder = os.path.join(*[RDSS_ROOT, 'preprocessed_data'])
print('Image paths', data_folder)

tifflist = glob.glob(data_folder + '/*/Z-Projection/*.tif')

print(len(tifflist), 'images found')

caselist = [x.split(os.path.sep)[-3] for x in tifflist]
caselist = list(set(caselist))
print(caselist)


result_folder = 'Detection_Results'
#adjust sample_microns if needed
sample_microns = 391.73

detection(data_folder, caselist, result_folder, sample_microns)

