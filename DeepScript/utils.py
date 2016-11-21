from __future__ import print_function

import glob
import random
import os
import copy
from itertools import product

SEED = 7687655
import numpy as np
np.random.seed(SEED)
random.seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from sklearn.feature_extraction.image import extract_patches_2d

from skimage import io
from skimage.transform import resize, downscale_local_mean
from scipy.misc import imsave

from keras.utils import np_utils

import augment

AUGMENTATION_PARAMS = {
    'zoom_range': (0.75, 1.25),
    'rotation_range': (-10, 10),
    'shear_range': (-15, 15),
    'translation_range': (-12, 12),
    'do_flip': False,
    'allow_stretch': False,
}

NO_AUGMENTATION_PARAMS = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def load_dir(dir_name, ext='.tif'):
    X, Y = [], []
    fns = glob.glob(dir_name+'/*' + ext)
    random.shuffle(fns)

    for fn in fns:
        img = np.array(io.imread(fn), dtype='float32')
        scaled = img / np.float32(255.0)
        scaled = downscale_local_mean(scaled, factors=(2,2))
        X.append(scaled)
        Y.append(os.path.basename(fn).split('_')[0])

    return X, Y

def augment_test_image(image, nb_rows, nb_cols, nb_patches):
    X = []
    patches = extract_patches_2d(image=image,
                                 patch_size=(nb_rows, nb_cols),
                                 max_patches=nb_patches)
    
    for patch in patches:
        patch = augment.perturb(patch, NO_AUGMENTATION_PARAMS, target_shape=(nb_rows, nb_cols))
        patch = patch.reshape((1, patch.shape[0], patch.shape[1]))
        X.append(patch)
    
    return np.array(X, dtype='float32')


def augment_train_images(images, categories, nb_rows, nb_cols, nb_patches):
    print('augmenting train!')
    X, Y = [], []
    for idx, (image, category) in enumerate(zip(images, categories)):
        if idx % 500 == 0:
            print('   >', idx)
        patches = extract_patches_2d(image=image,
                                     patch_size=(nb_rows * 2, nb_cols * 2),
                                     max_patches=nb_patches)
        for patch in patches:
            patch = augment.perturb(patch, AUGMENTATION_PARAMS, target_shape=(nb_rows, nb_cols))
            patch = patch.reshape((1, patch.shape[0], patch.shape[1]))
            X.append(patch)
            Y.append(category)

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='int8')

    return X, Y


def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=6)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


