# fn_playground

## Background
-----
`Fashion-MNIST` is a dataset of [Zalando](https://github.com/zalandoresearch/fashion-mnist)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.  `Fashion-MNIST` is intended to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

In this repo, I try to do some experiments with `Fashion-MNIST` dataset.

## Training Schedule
-----
1.  Standard preprocessing (mean/std subtraction/division) and data augment(Rand augment, random crops,horizontal flips, random erasing);
2.  Backbone: wide-resnet 40-4;
3.  Learning rate schedule: `CosineAnnealingLR`;
4.  Other tricks: Label smoothing, Exponential Moving Average, and so on.
## Usage
-----

1.run ```pip install -r requirements.txt```;<br><br>
2.run ```python fn_fmix_40_4_gn_ws_learning.py```;(Test on Windows10, Python 3.6 with GTX 1660. Best `Fashion-MNIST` test accuracy is 96.44%(epoch 562). log file `fn_fmix_40_4_gn_ws_600.txt` can be found in `models` dir. Model files can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1JfyBisN1kubm2rC_hHblai8MdNWkZxHU?usp=sharing))<br><br>
3.run ```python fn_fmix_40_4_bn_mish_ws_gem.py```;(Test on `Google Colab`. Best `Fashion-MNIST` test accuracy is 96.69%(epoch 881). log file `fn_fmix_40_4_bn_mish_ws_gem.txt` can be found in `models` dir. Model files can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1a6ZChTZJERsJp98mWy37kTHhRGOgT0mO?usp=sharing))<br><br>
4.run ```kaggle_cifar10_fmix_40_4_bn_mish_ws_gem.py```;(Test on `kaggle`. Best `Cifar-10` test accuracy is 98.03%(epoch 864). log file `cifar10_fmix_40_4_bn_mish_ws_gem.py.txt` can be found in `models` dir. Model files can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1BUYrTWH7_DGAGFSRdI7NLFmDMW5W716Q?usp=sharing))<br><br>
<table>
<thead><tr><th>script name</th><th>dataset</th><th>best test accuracy</th><th>weight files</th></tr><th>comments</th></tr></thead>
        <tr>
            <td><a href="">python fn_fmix_40_4_gn_ws_learning.py</a></td>
            <td><a >Fashion-MNIST</a></td>
            <td><a >96.44%(epoch 562)</a></td>
            <td><a href="https://drive.google.com/drive/folders/1JfyBisN1kubm2rC_hHblai8MdNWkZxHU?usp=sharing">Google Cloud</a></td>
  
        </tr>

</table>
