# fn_playground

## Background
-----
`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

In this repo, I try to do some experiments on `Fashion-MNIST` dataset.
## Training Schedule
-----
1.  Standard preprocessing (mean/std subtraction/division) and data augment(Rand augment, random crops,horizontal flips, random erasing);
2.  Backbone: wide resnet 40-4;
3.  Learning rate schedule: `CosineAnnealingLR`;
4.  Other tricks: Label smoothing, Exponential Moving Average, and so on.
## Usage
-----

1.run ```pip install -r requirements.txt```;<br><br>
2.run ```python fn_fmix_40_4_gn_ws_learning.py```;(Test on Windows10, Python 3.6 with GTX 1660. Best accuracy is 96.44%(epoch 562). log file `fn_fmix_40_4_gn_ws_600.txt` can be found in `models` dir. Model files can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1_JaYkBC-7KmewVHy5XFPzmJ0YnKziRIF?usp=sharing))<br><br>
2.run ```python fn_fmix_40_4_bn_mish_ws_gem.py```;(Test on `Google Colab`. Best accuracy is 96.69%(epoch 881). log file `fn_fmix_40_4_bn_mish_ws_gem.txt` can be found in `models` dir. Model files can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1a6ZChTZJERsJp98mWy37kTHhRGOgT0mO?usp=sharing))<br><br>
