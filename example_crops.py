from __future__ import print_function

import os
import shutil

import numpy as np

from skimage import io
from skimage.transform import downscale_local_mean

import scipy

from DeepScript import utils

NB_PATCHES = 30
NB_ROWS, NB_COLS = 150, 150
#FILEPATH = 'data/CLaMM_train/btv1b8451637v_f23.tif'
FILEPATH = 'data/CLaMM_train/IRHT_P_000046.tif'

try:
    os.mkdir('viz')
except:
    pass

example_dir = os.path.splitext(os.path.basename(FILEPATH))[0]

try:
    shutil.rmtree('viz/' + example_dir)
except:
    pass
os.mkdir('viz/' + example_dir)

img = io.imread(FILEPATH)

img = np.array(io.imread(FILEPATH), dtype='float32')
print(img.shape)
scipy.misc.imsave('viz/' + example_dir + '/orig.tiff', img)

scaled = img / np.float32(255.0)
scaled = downscale_local_mean(scaled, factors=(2,2))

scipy.misc.imsave('viz/' + example_dir + '/downscaled' + '.tiff', scaled * 255)


train_crops, _ = utils.augment_train_images([scaled], [0], NB_ROWS, NB_COLS, NB_PATCHES)
for idx, x in enumerate(train_crops):
    x = x.reshape((x.shape[1], x.shape[2]))
    x *= 255
    scipy.misc.imsave('viz/' + example_dir + '/train_crop_' + str(idx) + '.tiff', x)

test_crops = utils.augment_test_image(scaled, NB_ROWS, NB_COLS, NB_PATCHES)
for idx, x in enumerate(test_crops):
    x = x.reshape((x.shape[1], x.shape[2]))
    x *= 255
    scipy.misc.imsave('viz/' + example_dir + '/test_crop_' + str(idx) + '.tiff', x)




