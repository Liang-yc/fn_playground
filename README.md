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
2.choose a python script to run.<br><br>
<table  width="100%"  style="table-layout:fixed;word-break: break-all;  word-wrap:break-word;">
<thead><tr><th>script name</th><th>dataset</th><th>best test accuracy</th><th>weight files</th></tr></thead>
        <tr>
            <td><a href="https://github.com/Liang-yc/fn_playground/blob/master/fn_fmix_40_4_gn_ws_learning.py">fn_fmix_40_4_gn_ws_learning.py</a></td>
            <td><a >Fashion-MNIST</a></td>
            <td><a >96.44%(epoch 562)</a></td>
            <td><a href="https://drive.google.com/drive/folders/1JfyBisN1kubm2rC_hHblai8MdNWkZxHU?usp=sharing">Google Cloud</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/Liang-yc/fn_playground/blob/master/fn_fmix_40_4_bn_mish_ws_gem.py">fn_fmix_40_4_bn_mish_ws_gem.py</a></td>
            <td><a >Fashion-MNIST</a></td>
            <td><a >96.69%(epoch 881)</a></td>
            <td><a href="https://drive.google.com/drive/folders/1a6ZChTZJERsJp98mWy37kTHhRGOgT0mO?usp=sharing">Google Cloud</a></td>
        </tr>        
         <tr>
            <td><a href="https://github.com/Liang-yc/fn_playground/blob/master/kaggle_cifar10_fmix_40_4_bn_mish_ws_gem.py">kaggle_cifar10_fmix_40_4_bn_mish_ws_gem.py</a></td>
            <td><a >Cifar-10</a></td>
            <td><a >98.03%(epoch 864)</a></td>
            <td><a href="https://drive.google.com/drive/folders/1BUYrTWH7_DGAGFSRdI7NLFmDMW5W716Q?usp=sharing">Google Cloud</a></td>
        </tr>
</table>
