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
"""Generate synthetic background images from predefined tiles."""

from _common import _tile_paths, _background_dir
import os
import numpy as np
import utils.tiff
from tqdm import tqdm

def generate_backgrounds(n, shape=(512,512), tile_overlap_width=20):
    """Scissors cell images along the boundary contour.
    Arguments:
        n: a positive integer. number of background images to be generated.
        shape: a tuple/list of two positive integers. (optional)
        tile_overlap_width: a non-zero integer. overlap between tiles.
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
    height, width = (shape[0], shape[1])   # Size of the background
    tile_h, tile_w = (64, 64)       # Size of the tile
    tile_sampling_freq = 10         # From one tile source image, we sample multiple tiles
    
    # When placing tiles, there is going to be an overlap between the tiles.
    # The tile images will be blended in this overlap region for smooth transition.
    mask_transition = np.linspace(0, 1, tile_overlap_width)
    mask = np.ones((tile_h, tile_w))
    mask[:, 0:tile_overlap_width] = mask_transition
    mask[:, -tile_overlap_width:] = 1-mask_transition
    mask[0:tile_overlap_width, :] *= np.expand_dims(mask_transition, axis=1)
    mask[-tile_overlap_width:, :] *= np.expand_dims(1-mask_transition, axis=1)
    
    # sample tiles
    tiles = []
    for path in _tile_paths:
        img = utils.tiff.load_tiff(path)
        for j in range(0, img.shape[0]-tile_h, tile_sampling_freq):
            for i in range(0, img.shape[1]-tile_w, tile_sampling_freq):
                roi = img[j:j+tile_h, i:i+tile_w]
                tiles.append(roi - np.mean(roi))
    
    print('Generating synthetic backgrounds...')
    for cnt in tqdm(range(n)):
        background = np.zeros((height, width))
        for j in range(0, height, tile_h-tile_overlap_width):
            for i in range(0, width, tile_w-tile_overlap_width):
                imax = min(i+tile_w, width-1)
                jmax = min(j+tile_h, height-1)
                idx = np.random.randint(0, len(tiles))
                temp = tiles[idx]*mask
                background[j:jmax, i:imax] += temp[0:jmax-j, 0:imax-i]
    
        background += np.random.normal(150.0, 80.0) # brightness purturbation
        background += np.random.normal(0.0, 20.0, (height, width)) # noise

    
        utils.tiff.save_tiff(background, os.path.join(*[_background_dir, '%08d.tif'%cnt]))