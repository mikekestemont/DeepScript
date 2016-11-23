# DeepScript

## Introduction
This repository holds the code which was developed for the [ICFHR2016 Competition on the Classification of Medieval Handwritings in Latin Script](http://oriflamms.hypotheses.org/1388), of which the results are described in the following paper:

> Florence Cloppet, Véronique Eglin, Van Cuong Kieu, Dominique Stutzmann, Nicole Vincent. 'ICFHR2016 Competition on the Classification of Medieval Handwritings in Latin Script'. In: *Proceedings of the 15th International Conference on Frontiers in Handwriting Recognition* (2016), 590-595. DOI 10.1109/ICFHR.2016.106.

The main task for this competition was to correctly predict the script type of samples of medieval handwriting, as represented by single-page, grescale photographic reproductions of codices. A random selection of examples goes below:

![Random examples](https://cloud.githubusercontent.com/assets/4376879/20482409/d5d13558-afec-11e6-8350-b4b45cc991f7.png "Random examples")


The 'DeepScript' approach described here scored best in the second task for this competition ("fuzzy classification"). This system uses a ‘vanilla’ neural network model, i.e. a single-color channel variation of the popular [VGG-architecture](https://arxiv.org/abs/1409.1556). The model takes the form of a stack of convolutional layers, each with a 3x3 perceptive field an increasingly large number of filters at each block of layers (2 x 64 > 3 x 128 > 3 x 256). This convolutional stack feeds into two fully-connected dense layers with a dimensionality of 1048, before feeding into the final softmax layer where the normalized scores for each class label get predicted. Because of the small size of the datasets, DeepScript borrowed the augmentation from the work by [Sander Dieleman](http://benanne.github.io/about/) and colleagues which is described in this [awesome blog post](http://benanne.github.io/2015/03/17/plankton.html). The original code is available from [this repository](https://github.com/benanne/kaggle-ndsb). We would like to thank Sander for his informal help and advice on this project. Below goes an example of the augmented patches on which the model was trained (see `example_crops.py`):

![Examples of augmented patched](https://cloud.githubusercontent.com/assets/4376879/20482407/d5bd87e2-afec-11e6-9565-f53f8519c118.png "Examples of augmented patched")


## Scripts

The following top-level scripts included in the repository are useful at a higher level:
- `prepare_data.py`: prepare and preprocess train/dev/test sets of the original images
- `train.py`: train a new model
- `test.py`: test/apply a previously trained model
- `filter_viz.py`: visualize what filters were learned during training
- `example_crops.py`: generate examples of the sort of augmented crops used in training
- `crop_activations.py`: find out which patches from a test set maximally activate a particular class

By default, new models are stored in a directory under the `models` directory in the repository. A pretrained model can be downloaded as a ZIP archive from [Google Drive](https://drive.google.com/open?id=0B84gHUu-mY-IR0FGdnNNUjdFTVU): unzip it and place it under a `models` directory in the top-level directory of the repository. The original data can be obtained via registering on the competition's [website](http://oriflamms.hypotheses.org/1388).


## Visualizations

Of special interest was the ability to visualize the knowledge inferred by a trained neural network (i.e. the question: what does the network 'see' after training? Which sort of features has it become sensitive to?). For this visualization, we heavily drew from the excellent set of example scripts offered in the [keras library](https://keras.io/). Below, we show examples of randomly initialized images which were ajusted via the principle of gradient ascent to maximally activate single neurons on the final convolutional layer (see `filter_viz.py`). The generated images have been annotated with a couple of interesting paleographic features that seem to emerge:


![Visualization of filter activations](https://cloud.githubusercontent.com/assets/4376879/20482410/d5d99978-afec-11e6-98f8-0b057a77e9ba.png "Visualization of filter activations")

Additionally, it is possible to select the patches from the unseen test images which maximally activated the response of a certain class out the output layer. Examples of top-activating patches (without augmentation) are given below.

![Best test patches for classes](https://cloud.githubusercontent.com/assets/4376879/20482411/d5e2cb38-afec-11e6-9a38-5fd559bdc167.png "Best test patches for classes")

The confusion matrix obtained for the development data shows that the model generally makes solid predictions, and mostly makes understandable errors (e.g. the confusion between different types of *textualis* variants):

![Confusion matrix validation set](https://cloud.githubusercontent.com/assets/4376879/20482408/d5bd74dc-afec-11e6-894b-6485144c1ed0.png "Confusion matrix validation set")


## Dependencies

Major dependencies for this code include:

- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [numpy](http://www.numpy.org/)
- [theano](http://deeplearning.net/software/theano/)
- [keras](https://keras.io/)
- [h5py](http://www.h5py.org/)
- [scipy](https://www.scipy.org/) 
- [scikit-image](http://scikit-image.org/)
- [seaborn](http://seaborn.pydata.org/)
- [ghalton](https://pypi.python.org/pypi/ghalton)

These packages can be easily installed via [pip](https://pypi.python.org/pypi/pip). I recommend Continuum's [Anaconda environment](https://www.continuum.io/downloads) with ships with most of these dependencies. The code has been tested on Mac OX S and Linux Ubuntu under Python 2.7. Note that this version number might affect your ability to load some of the pretrained model's components as pickled objects.

## Publication
The preliminary results of our specific approach have been presented at the [2016 ESTS conference in Antwerp](https://textualscholarship.eu/ests-2016/). A dedicated publication describing our approach is on the way. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the TITAN X used for this research.

