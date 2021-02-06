from torch.utils.data import Dataset,DataLoader
import pickle
import warnings
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url
import os
import os.path
import gzip
import numpy as np
import torch
from PIL import Image
import codecs
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class CustomDataset(Dataset):#需要继承data.Dataset
    def __init__(self,val_data,val_label,transform):
        # TODO
        # 1. Initialize file path or list of file names.
        self.data = val_data
        self.targets = val_label
        self.transform = transform
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        print(e)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class MyDataset(Dataset):
    def __init__(self, filepath, transform=None,keys = None, target_transform=None):
        with open(filepath,'rb') as f:
            self.data = pickle.load(f)
        self.keys = keys
        self.input_seq = self.data[self.keys[0]]  ### 输入序列
        self.output_seq = self.data[self.keys[1]]   #### 输出序列
        self.transform = transform #### 对输入序列进行变换
        self.target_transform = target_transform   ###### 对输出序列进行变换

    def __getitem__(self, index):
        input_seq,output_seq = self.input_seq[index],self.output_seq[index]  ## 按照索引迭代读取内容
        if self.transform is not None:
            input_seq = self.transform(input_seq)
            output_seq = self.transform(output_seq)
        return input_seq,output_seq  ### 直接输出输入序列和输出序列

    def __len__(self):
        return self.data[self.keys[0]].shape[0]   ### 返回的是样本集的大小，样本的个数



class fashionmnist(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # urls = [
    #     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    # ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    # classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
    #            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,split_id_range=[-1,-1],need_split=True):
        super(fashionmnist, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if self.train:
            all_data,all_label = torch.load(os.path.join(self.processed_folder, data_file))
            if need_split:  #选取的时训练集的一部分
                self.data = torch.cat([all_data[:split_id_range[0],:,:],all_data[split_id_range[1]:,:,:]])
                self.targets = torch.cat([all_label[:split_id_range[0]],all_label[split_id_range[1]:]])
            else:
                self.data= all_data[split_id_range[0]:split_id_range[1],:,:]
                self.targets= all_label[split_id_range[0]:split_id_range[1]]
        else:

            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class fashionmnist_addon(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # urls = [
    #     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    # ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    # classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
    #            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,idx_list=[],need_delete=True,double=False):
        super(fashionmnist_addon, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if need_delete:
            print("del ",len(idx_list))
            idx = np.arange(self.data.shape[0])
            idx = np.delete(idx,idx_list,axis=0)
            self.data = self.data[idx, :, :]
            self.targets = self.targets[idx]
        else:
            self.data = torch.cat((self.data,self.data[idx_list,:,:]),dim=0)
            self.targets = torch.cat((self.targets,self.targets[idx_list]),dim=0)

        if double:
            self.data = torch.cat((self.data, torch.flip(self.data, [1, 2])), dim=0)
            self.targets = torch.cat((self.targets, self.targets), dim=0)

        # self.data.extend(self.data[idx_list,:,:,:])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import csv


#
class MNIST_t(data.Dataset):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    train_triplet_file = 'train_triplets.txt'
    test_triplet_file = 'test_triplets.txt'

    def __init__(self, root, n_train_triplets=60000, n_test_triplets=10000, train=True, transform=None,
                 target_transform=None, download=False):
        self.root = root

        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            self.make_triplet_list(n_train_triplets)
            triplets = []

            for line in open(os.path.join(root, self.processed_folder, self.train_triplet_file)):

                # print(line)
                if line.strip('\n'):
                    triplets.append(
                        (int(line.split()[0]), int(line.split()[1]), int(line.split()[2])))  # anchor, close, far
            self.triplets_train = triplets
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))
            self.make_triplet_list(n_test_triplets)
            triplets = []
            for line in open(os.path.join(root, self.processed_folder, self.test_triplet_file)):
                # print(line)
                if line.strip('\n'):
                    triplets.append(
                        (int(line.split()[0]), int(line.split()[1]), int(line.split()[2])))  # anchor, close, far
            self.triplets_test = triplets

    def __getitem__(self, index):
        if self.train:
            idx1, idx2, idx3 = self.triplets_train[index]
            img1, img2, img3 = self.train_data[idx1], self.train_data[idx2], self.train_data[idx3]
        else:
            idx1, idx2, idx3 = self.triplets_test[index]
            img1, img2, img3 = self.test_data[idx1], self.test_data[idx2], self.test_data[idx3]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        if self.train:
            return len(self.triplets_train)
        else:
            return len(self.triplets_test)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_triplets_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_triplet_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_triplet_file))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def make_triplet_list(self, ntriplets):

        if self._check_triplets_exists():
            return
        print('Processing Triplet Generation ...')
        if self.train:
            np_labels = self.train_labels.numpy()
            filename = self.train_triplet_file
        else:
            np_labels = self.test_labels.numpy()
            filename = self.test_triplet_file
        triplets = []
        for class_idx in range(10):
            a = np.random.choice(np.where(np_labels == class_idx)[0], int(ntriplets / 10), replace=True)
            b = np.random.choice(np.where(np_labels == class_idx)[0], int(ntriplets / 10), replace=True)
            while np.any((a - b) == 0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels != class_idx)[0], int(ntriplets / 10), replace=True)

            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(c[i]), int(b[i])])

        with open(os.path.join(self.root, self.processed_folder, filename), "w") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(triplets)
        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

if __name__ == '__main__':
    # import torch.utils.data as data
    # import cv2
    # data, targets = torch.load('./data/FashionMNIST/processed/training.pt')
    # train = data.numpy()
    # train_label = targets.numpy()
    # path='./fn'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # for i in range(len(train)):
    #     print(i,len(train))
    #     img = train[i,:,:]
    #     save_path = path+'/'+str(i)+'.png'
    #     cv2.imwrite(save_path,img)
    #
    # from imagededup.methods import CNN
    # cnn_encoder = CNN()
    # duplicates = cnn_encoder.find_duplicates(image_dir='./fn',
    #                                          # max_distance_threshold=12,
    #                                          min_similarity_threshold=0.9,
    #                                          scores=True,
    #                                          outfile='./my_duplicates.json')
    import numpy as np
    dum_list = np.load("./dum_List.npy")

    f = open('F:/lyc/test/my_duplicates.json')
    json_data = json.load(f)
    # dum_mat = np.zeros(shape=(60000,60000),dtype=np.uint8)
    non_list = []
    for i in range(60000):
        img_name = str(i)+'.png'
        current_sample = json_data[img_name]
        for j in range(len(current_sample)):
            current_img_name = current_sample[j][0]
            current_id = int(current_img_name[:current_img_name.rfind(".png")])
            if i<current_id and float(current_sample[j][1])>0.95:
                # dum_mat[i,current_id]=1
                non_list.append([i,current_id])

    non_list = sorted(non_list,key=lambda a:a[0])
    print('list =',len(non_list))
    dum_list=[]
    #每行先记录下来，如果有重合，就合并
    current_id =-1
    for i in range(len(non_list)):
        if non_list[i][0]==current_id:
            dum_list[len(dum_list)-1].append(non_list[i][1])
        else:
            # print(i,len(non_list))
            current_id=non_list[i][0]
            dum_list.append([])
            dum_list[len(dum_list)-1].append(non_list[i][0])
            dum_list[len(dum_list)-1].append(non_list[i][1])
    # for i in range(len(non_list)):
    #     print(i,len(non_list))
    #     j=0
    #     while(j<len(dum_list)):
    #         if non_list[i][0]==dum_list[j][0]:
    #             dum_list[j].append(non_list[i][1])
    #             break
    #         j+=1
    #     if j==len(dum_list):
    #         dum_list.append([])
    #         dum_list[len(dum_list)-1].append(non_list[i][0])
    #         dum_list[len(dum_list)-1].append(non_list[i][1])

    print(len(dum_list))
    for i in range(len(dum_list)):
        if len(dum_list[i]) < 1:
            continue
        for j in range(i+1,len(dum_list)):
            if len(dum_list[j]) < 1:
                continue
            if dum_list[i][len(dum_list[i])-1]<dum_list[j][0]:
                continue
            matched=False
            for m in range(len(dum_list[i])):
                for n in range(len(dum_list[j])):
                    if dum_list[i][m]==dum_list[j][n]:
                        matched=True
                        break

                if matched:
                    break
            #<class 'list'>: [38, 13311, 34347, 38754, 53768]
            if matched:
                print(i,j)

                for n in range(len(dum_list[j])):
                    m=0
                    while m<len(dum_list[i]):
                        if dum_list[i][m] == dum_list[j][n]:
                            break
                        m+=1
                    if m==len(dum_list[i]):
                        dum_list[i].append(dum_list[j][n])
                dum_list[i] = sorted(dum_list[i])
                dum_list[j]=[]


    print(len(dum_list))
    num=0
    for i in  range(len(dum_list)):
        if len(dum_list[i])<1:
            continue
        num+=len(dum_list[i])
        print(dum_list[i])
    print('num=',num)
    import numpy as np
    np.save('./dum_List.npy',dum_list)
    a=1