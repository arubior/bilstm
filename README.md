# Learning Fashion Compatibility with Bidirectional LSTMs - PyTorch version
PyTorch implementation of the [paper](https://arxiv.org/pdf/1707.05691.pdf)

**INSTALLATION**
In order to run this code, you must install:
* [PyTorch](http://pytorch.org/previous-versions/) (install it with CUDA support if you want to use GPUs, which is strongly recommended).
* Python with packages numpy, torchvision, tensorboardX, PIL, collections, cv2 (only if you want to generate result images for fill in the blank task, tested with version 2.4.8), h5py, sklearn, nltk.

The code has been tested with these two configurations:
    * python 2.7.6 - Pytorch 0.3.0.post4     - CUDA 7.5 - numpy 1.11.1 - sklearn 0.18.1
    * python 3.6.0 - PyTorch 0.4.0a0+040336f - CUDA 9.1 - numpy 1.13.3 - sklearn 0.19.1

**TRAINING**
To train the model, run:
```
python main.py
```

You can take a look at the evolution of the training by running `tensorboard --logdir runs`.

Note that we included options to change the default parameters (learning rate, batch size, and convolutional model to extract image features). By default, these are just like in the paper: 0.2, 10 and inception_v3.

**EVALUATION**
Prior to running the evaluation of the models, you must extract the features of the test images.
We provide a snapshot for our trained model in _models/pretrained_.
In order to do so, run:
```
python src/get_features.py -m model_path -sp path_to_save_data [-t model_type]
```
This will save a h5 file with the image features.

***Evaluation of compatibility AUC***
To reproduce the results in the 2nd column of table 1 of the paper, run:
```
python src/evaluation.py -m model_path -sp path_to_saved_data [-t model_type]
```

***Evaluation of fill in the blank accuracy***
To reproduce the results in the 1st column of table 1 of the paper, run:
```
python src/outfit_generation.py -m model_path -sp path_to_saved_data -i path_to_save_results[-t model_type]
```
