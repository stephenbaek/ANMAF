# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:04:14 2019

@author: sbaek
"""
import os
from _common import root_dir, _tile_paths, _background_dir
from _background_gen import generate_backgrounds
from _image_synth import synthesize_images
import argparse
import shutil
import pathlib

###############################################################################
# PARSE COMMAND-LINE ARGUMENTS

parser = argparse.ArgumentParser(description='Generate synthetic datasets for cell segmentation.')
#parser.add_argument('num_images', metavar='NUM_IMAGES', type=int, nargs=1,
#                    help='number of images to be generated')
parser.add_argument('-n', '--num_images', dest='num_images', type=int, nargs='?', default=1000,
                   help='number of images to be generated (default: 1,000)')
parser.add_argument('-b', '--backgrounds', dest='num_backgrounds', type=int, nargs='?', default=300,
                   help='number of background images to be used (default: 300)')
parser.add_argument('-d', '--dir', dest='directory', type=str, nargs='?', default='train',
                   help='directory name where the images will be generated')

args = parser.parse_args()
###############################################################################


###############################################################################
# GENERATE BACKGROUNDS

# File paths to synthetic backgrounds
_background_paths = [str(path) for path in list(pathlib.Path(_background_dir).glob('*.tif'))]

if not len(_background_paths) == 0:
    while True:
        query = input('[Warning] Background images already exist. Would you like to discard and regenerate them all? [y/n] ')
        key = query[0].lower()
        if query == '' or not key in ['y','n']:
            print('Please answer with yes or no! ')
        else:
            break
    if key == 'y':
        for path in _background_paths:
            os.remove(path)
        generate_backgrounds(args.num_backgrounds)
    if key == 'n':
        #DO nothing
        print('Skipping background generation.')
else:
    generate_backgrounds(args.num_backgrounds)
###############################################################################



###############################################################################
# SYNTHESIZE IMAGES
path = os.path.join(*[root_dir, 'data', args.directory])
if os.path.isdir(path):
#    print(args.directory + ' exists. Images will be overwritten. Would you like to continue? [y/n]')
    while True:
        query = input('[Warning] Directory `' + args.directory + '` already exists. Images will be overwritten. Would you like to continue? [y/n] ')
        key = query[0].lower()
        if query == '' or not key in ['y','n']:
            print('Please answer with yes or no! ')
        else:
            break
    if key == 'y':
        shutil.rmtree(path)
        os.mkdir(path)
    if key == 'n':
        exit(0)
else:
    os.mkdir(path)
synthesize_images(args.num_images, path)
###############################################################################

print('Synthetic image generation successful!')