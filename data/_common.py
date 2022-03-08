import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.dirname(os.path.abspath(__file__))


sys.path.append(root_dir)
import config
import pathlib

# File paths to cell annotations (the ImageJ ROI files)
_rdss_root = config.RDSS_ROOT
_roi_paths = [str(path) for path in list(_rdss_root.glob('*/*.zip'))]

_image_dir = 'Z-Projection'

_background_dir = os.path.join(*[data_dir, 'background'])

# File paths to background tiles
_tile_paths = pathlib.Path(os.path.join(*[_background_dir, 'tiles']))
_tile_paths = [str(path) for path in list(_tile_paths.glob('*.tif'))]

