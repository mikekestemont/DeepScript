"""
Module to preprocess the CLaMM data:
    - create a train/dev split from the training data and preprocess these
    - preprocess the test images
"""


from __future__ import print_function

import os
import shutil

SEED = 1066987
import random
import numpy as np

np.random.seed(SEED)
random.seed(SEED)

from sklearn.cross_validation import train_test_split
    

def train_dev_split(input_dir = 'data/CLaMM_train',
                    metafile = 'meta.csv',
                    target_classes = None,
                    test_size = 0.1,
                    random_state = 863863,
                    output_path = None):

    print('Creating train / dev split for data from:', input_dir)
    if target_classes:
        print('\t-> restricting to classes:', target_classes)

    filenames, categories = [], []

    for line in open(os.sep.join((input_dir, metafile)), 'r'):

        line = line.strip()

        if line and not line.startswith('FileName'):
            filename, category = line.split(';')
            category = category.lower().replace('_', '-')

            if target_classes:

                if category not in target_classes:
                    continue

                if not os.path.exists(os.sep.join((input_dir, filename))):
                    raise ValueError('%s not found!' % (filename))

            filenames.append(filename)
            categories.append(category)

    print('categories:', sorted(set(categories)))

    assert len(filenames) == len(categories)
    print('\t-> splitting', len(filenames), 'items in total')

    train_fns, dev_fns, train_categories, dev_categories = \
            train_test_split(filenames, categories,
                             test_size=test_size,
                             random_state=random_state,
                             stratify=categories)

    print('\n# train images:', len(train_fns))
    print('# dev images:', len(dev_fns))

    os.mkdir(os.sep.join((output_path, 'train')))
    os.mkdir(os.sep.join((output_path, 'dev')))

    for fn, category in zip(train_fns, train_categories):
        in_ = os.sep.join((input_dir, fn))
        out_ = os.sep.join((output_path, 'train', category + '_' + fn))
        shutil.copyfile(in_, out_)

    for fn, category in zip(dev_fns, dev_categories):
        in_ = os.sep.join((input_dir, fn))
        out_ = os.sep.join((output_path, 'dev', category + '_' + fn))
        shutil.copyfile(in_, out_)

def prepare_test_data(input_dir = 'data/CLaMM_test',
                      metafile = 'meta.csv',
                      target_classes = None,
                      output_path = None):
    
    print('Preprocessing test data available under:', input_dir)
    if target_classes:
        print('\t-> restricting to classes:', target_classes)

    filenames, categories = [], []

    for line in open(os.sep.join((input_dir, metafile)), 'r'):

        line = line.strip()

        if line and not line.startswith('FileName'):
            filename, category = line.split(';')
            category = category.lower().replace('_', '-')

            if target_classes:

                if category not in target_classes:
                    continue

                if not os.path.exists(os.sep.join((input_dir, filename))):
                    raise ValueError('%s not found!' % (filename))

            filenames.append(filename)
            categories.append(category)

    print('categories:', sorted(set(categories)))

    assert len(filenames) == len(categories)

    print('\n# test images:', len(filenames))

    os.mkdir(os.sep.join((output_path, 'test')))

    for fn, category in zip(filenames, categories):
        in_ = os.sep.join((input_dir, fn))
        out_ = os.sep.join((output_path, 'test', category + '_' + fn))
        shutil.copyfile(in_, out_)


if __name__ == '__main__':
    
    # which classes?
    target_classes = None # e.g. ['caroline', 'textualis']

    # create path for the train, dev, test splits
    output_path = os.path.dirname(os.path.realpath(__file__))+'/data/splits'

    try:
        shutil.rmtree(output_path)
    except:
        pass

    os.mkdir(output_path)

    # create train-dev split:
    train_dev_split(input_dir='data/CLaMM_train',
                    test_size = .1,
                    random_state = SEED,
                    output_path = output_path,
                    target_classes = target_classes)

    # preprocess test data:
    prepare_test_data(input_dir='data/CLaMM_test',
                      output_path = output_path,
                      target_classes = target_classes)

