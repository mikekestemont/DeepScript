'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
import shutil
import os
import glob
import h5py
import pickle

from keras import backend as K
from keras.models import model_from_json, Sequential
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import numpy as np

from skimage import io
from skimage.transform import downscale_local_mean
import scipy

from DeepScript import utils

# dimensions of the generated pictures for each filter.
img_width = 150
img_height = 150

# the name of the layer we want to visualize (see model definition below)
MODEL_NAME = 'final'
NB_FILTERS = 12
NB_TEST_PATCHES = 100
#LAYER_NAME = 'conv4_3'
LAYER_NAME = 'prediction'
NB_TOP_PATCHES = 5

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


model = Sequential()
model.add(ZeroPadding2D((1,1), input_shape=(1, img_width, img_height)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', name='fc1'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax', name='prediction'))
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


#model = model_from_json(open('models/' + MODEL_NAME + '/architecture.json').read())
model.load_weights('models/' + MODEL_NAME + '/weights.hdf5')

input_img = model.input

print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

if os.path.isdir('crop_viz'):
    shutil.rmtree('crop_viz')
os.mkdir('crop_viz')

test_crops = []
for fn in glob.glob('data/splits/dev/*.tif'):
    img = None
    img = np.array(io.imread(fn), dtype='float32')
    scaled = None
    scaled = img / np.float32(255.0)
    scaled = downscale_local_mean(scaled, factors=(2,2))
    test_crops.extend(utils.augment_test_image(scaled, img_width, img_height, NB_TEST_PATCHES))

print(len(test_crops))

if LAYER_NAME == 'prediction':
    label_encoder = pickle.load( open('models/' + MODEL_NAME + '/label_encoder.p', 'rb'))
    print('-> Working on classes:', label_encoder.classes_)
    NB_FILTERS = len(label_encoder.classes_)

for filter_idx in range(0, NB_FILTERS):
    print('Processing filter', filter_idx)
        
    layer_output = layer_dict[LAYER_NAME].output
    if LAYER_NAME == 'prediction':
        loss = K.mean(layer_output[:, filter_idx])
    else:
        loss = K.mean(layer_output[:, filter_idx, :, :])
        
    filter_activation = K.function([input_img, K.learning_phase()], [loss])

    test_activations = []
    for img in test_crops:
        test_activations.append(filter_activation([[img], 0]))

    test_activations = np.array(test_activations, dtype='float32')

    os.mkdir('crop_viz/filter_'+str(filter_idx))
    top_idxs = test_activations.ravel().argsort()[::-1][:NB_TOP_PATCHES]

    for top_idx in top_idxs:
        top_img = None
        top_img = test_crops[top_idx]
        top_img = top_img.reshape((top_img.shape[1], top_img.shape[2]))
        top_img *= 255
        scipy.misc.imsave('crop_viz/filter_'+str(filter_idx)+'/'+str(top_idx)+'.tiff', top_img)

