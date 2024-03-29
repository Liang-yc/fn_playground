# 原始代码


# Hi, here:
# John: use LocalFashionMNIST to avoid download fashion behind proxy.  1)mkdir ./fashion/raw  2)download data and unzip to raw  3)mkdir ./fashion/processed  4)program tidy it


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from RandAugment import RandAugment
from autoaugment import RandAugment
import time

# from bi_loss import bi_tempered_logistic_loss
t1 = 0.7
t2 = 1.3
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt

dir_path = '/kaggle/working/models/'  # 'fn_fmix_40_4_bn_mish_ws_gem_kd'
save_path = dir_path + '/fn_fmix_40_4_bn_mish_ws_gem.pkl'
best_path = dir_path + '/best_' + 'fn_fmix_40_4_bn_mish_ws_gem.pkl'
data_path = './data'
ema_path = dir_path + '/ema_' + 'fn_fmix_40_4_bn_mish_ws_gem.pkl'
log_txt = dir_path + '/fn_fmix_40_4_bn_mish_ws_gem.txt'
import torch
import torch.nn.functional as F
from torch import nn
import torchcontrib
from my_Loss import batch_hard_triplet_loss, CenterLoss

margin = 0.3
epoch_nb = 1000


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# Code for Weight Standardization
class Conv2d(nn.Conv2d):
    def __init__(self, in_planes, planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_planes, planes, kernel_size, stride,
                         padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class gray2rgb(object):
    def __call__(self, img):
        enh_contrast = img.convert("RGB")

        # enh_contrast =
        return enh_contrast


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, my_conv=Conv2d):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.bn1 = nn.GroupNorm(num_groups=8,num_channels=in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = Mish()
        self.conv1 = my_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # self.bn2 = nn.GroupNorm(num_groups=8,num_channels=out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = Mish()
        self.conv2 = my_conv(out_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)
        self.droprate = dropRate
        from dropblock import LinearScheduler, DropBlock2D
        self.conv2_drop = LinearScheduler(
            DropBlock2D(drop_prob=dropRate, block_size=3),
            start_value=0,
            stop_value=dropRate,
            nr_steps=10000
        )

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and my_conv(in_planes, out_planes, kernel_size=1, stride=stride,
                                                              padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            # x = self.bn1(self.relu1(x))
        else:
            out = self.relu1(self.bn1(x))
            # out = self.bn1(self.relu1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # out = self.bn2(self.relu2(self.conv1(out if self.equalInOut else x)))

        if self.droprate > 0:
            #     out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2_drop(out)

        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.2, se_ratio=6, my_conv=Conv2d):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, my_conv)
        self.se_ratio = se_ratio

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, my_conv):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, my_conv))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        # if self.se_ratio:
        #     x_squeezed = F.adaptive_avg_pool2d(x, 1)
        #     x_squeezed = self._se_expand(self.relu(self._se_reduce(x_squeezed)))
        #     x = torch.sigmoid(x_squeezed) * x
        return x
        # return self.layer(x)


class SE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class TPJ(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.2, use_ws=False):
        super(TPJ, self).__init__()
        nChannels = [64, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        if use_ws:
            my_conv = Conv2d
        else:
            my_conv = nn.Conv2d
        self.conv1 = my_conv(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)  # John: first 3-->1
        self.conv2 = my_conv(nChannels[0], nChannels[0], kernel_size=3, stride=1, padding=1,
                             bias=False)  # John: first 3-->1
        self.conv3 = my_conv(nChannels[0], nChannels[0], kernel_size=3, stride=1, padding=1,
                             bias=False)  # John: first 3-->1

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, my_conv)

        # self.seblock1 = SE_Module(nChannels[1])

        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, my_conv)
        # self.seblock2 = SE_Module(nChannels[2])
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, my_conv)
        # self.seblock3 = SE_Module(nChannels[3])
        self.supervisor_cnum = nChannels[3]
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        # self.bn1 = nn.GroupNorm(num_groups=8,num_channels=nChannels[3])
        # self.relu = nn.ReLU(inplace=True)
        self.relu = Mish()

        from my_layers import GeM
        self.gem = GeM()

        self.center = nn.Linear(nChannels[3], nChannels[3])
        self.bnneck = nn.BatchNorm1d(nChannels[3])
        self.bnneck.bias.requires_grad_(False)
        self.fc = nn.Linear(nChannels[3], num_classes, False)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if 0 and isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #                 m.weight.data.normal_(0, math.sqrt(2. / n))#这个好
                nn.init.kaiming_normal_(m.weight.data)  # 卷积层参数初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.block1(out)
        # out = self.seblock1(out)
        out = self.block2(out)
        # out = self.seblock2(out)
        out = self.block3(out)
        # out = self.seblock3(out)
        # feature = self.bn1(self.relu(out))
        feature = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 7)
        # out = F.adaptive_avg_pool2d(feature, 1)
        out = self.gem(feature)
        out = out.view(-1, self.nChannels)
        center = self.center(out)
        # out = center
        out = self.bnneck(center)
        # if self.training:
        #     return self.fc(out),center
        return self.fc(out)


import random
import math

from PIL import Image


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class gray2rgb(object):
    def __call__(self, img):
        enh_contrast = img.convert("RGB")

        # enh_contrast =
        return enh_contrast


class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914, 0.4914, 0.4914]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


def accuracy(output,
             target):  # this 'accuracy' is equivalent to fashion's accuracy (Han Xiao, Fashion-Dataset's owner: https://github.com/zalandoresearch/fashion-mnist/issues/136)
    maxk = max((1,))
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].view(-1).float().sum(0)
    return correct_k, target.size(0)


def swa_train(use_cuda, trainloader, testloader, learning_rate_init):
    learning_rate = learning_rate_init
    t1, t2 = 0.8, 1.2
    start_epoch = 0

    best_acc = 0

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print('best loaded')
        best_acc = test(use_cuda, testloader)  # 先测一遍不就好了

    if os.path.exists(ema_path):
        checkpoint = torch.load(ema_path)
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print('ema loaded')

    print("inject ema")
    ema = EMA(model, 0.999)
    ema.register()
    print("go on", start_epoch)
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print('no ema loaded,load epoch weight')
    else:
        print('new train')

    #     best_acc = test(use_cuda, testloader)  # 先测一遍不就好了
    # best_acc = test_tta(use_cuda,valloader)

    criterion = LabelSmoothing()
    ce_criterion = torch.nn.CrossEntropyLoss()
    warm_up_epochs = 10
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * (math.cos((epoch - warm_up_epochs) / (epoch_nb - warm_up_epochs) * math.pi) + 1)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=warm_up_with_cosine_lr)
    #                                                     last_epoch =start_epoch)
    print('start_epoch:', start_epoch)
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_nb)
    #
    if start_epoch > 0:
        #     print(start_epoch, optimizer.param_groups[0]['lr'])
        lr_scheduler.step(start_epoch)
    #     print(start_epoch, optimizer.param_groups[0]['lr'])

    for epoch in range(start_epoch, epoch_nb + 1):
        model.train()
        correctAll = 0
        totalAll = 0
        lr_scheduler.step(epoch)
        seed_torch(seed=42+epoch)
        for batch, (input, target) in enumerate(trainloader):
            if use_cuda:
                input, target = input.cuda(), target.cuda()
            input, target = torch.autograd.Variable(input), torch.autograd.Variable(target)
            r = np.random.rand(1)
            if r > 0.2 and r < 1.0 and epoch>10: #cutmix
                # generate mixed sample
                lam = np.random.beta(0.2, 0.2)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                # print(use_cuda,target_a.is_cuda,target_b.is_cuda)
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                # compute output
                output = model(input)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)+ \
                       ce_criterion(output, target_a) * lam + ce_criterion(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if batch % 5 == 0:
                ema.update()
            correct, total = accuracy(output.data, target.data)
            correctAll += correct
            totalAll += total
            print('train:    batch=%04d' % batch, '  accuracy=%.2f' % (correct / total * 100.0), '  epoch=%04d' % epoch,
                  end='\r')

        print('TRAIN:    batch=%04d' % batch, 'lr=%04f'%(optimizer.state_dict()['param_groups'][0]['lr']),'  accuracy=%.2f' % (correctAll / totalAll * 100.0),
              '  epoch=%04d' % epoch)

        f = open(log_txt, 'a+')
        line = str(epoch) + "  " + str(batch) + "  lr " +str(optimizer.state_dict()['param_groups'][0]['lr'])+ "  "+str((correctAll / totalAll * 100.0)) + '\n'
        f.write(line)
        f.close()

        lr_scheduler.step(epoch)

        # ema.apply_shadow()


        torch.save({'epoch': epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': model.state_dict(), }, save_path)

        if epoch>epoch_nb//2:
            acc = test(use_cuda, testloader)
            if acc > best_acc:
                best_acc = acc
                torch.save({'epoch': epoch + 1,
                            # 'arch': args.arch,
                            'acc': acc,
                            'state_dict': model.state_dict(), }, best_path)
            ema.apply_shadow()
            torch.save({'epoch': epoch + 1,
                        # 'arch': args.arch,
                        'state_dict': model.state_dict(), }, ema_path)
            acc = test(use_cuda, testloader)

            if acc > best_acc:
                best_acc = acc
                torch.save({'epoch': epoch + 1,
                            # 'arch': args.arch,
                            'acc': acc,
                            'state_dict': model.state_dict(), }, best_path)
            ema.restore()


def test_only(use_cuda, testloader):
    model.eval()
    correctAll = 0
    totalAll = 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            pred = np.zeros(shape=(targets.shape[0], 10))
            label = targets.numpy()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            for i in range(targets.shape[0] // 100):
                outputs = model(inputs[i * 100:(i + 1) * 100, :, :, :])
                pred[i * 100:(i + 1) * 100, :] = outputs.data.cpu()
                # loss = criterion(outputs, targets) + mcloss(feature, targets) * 10
                # loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
                correct, total = accuracy(outputs.data, targets[i * 100:(i + 1) * 100].data)
                correctAll += correct
                totalAll += total
                print('using:    batch=%04d' % batch, '  accuracy=%.2f' % (correct / total * 100.0), end='\r')
        print('USING:    batch=%04d' % batch, '  accuracy=%.2f' % (correctAll / totalAll * 100.0))
    print('')
    pred = np.argmax(pred, 1)
    index = np.arange(pred.shape[0])
    mis_label = index[pred != label]
    return correctAll / totalAll * 100.0, mis_label


def test_tta(use_cuda, testloader):
    model.eval()
    correctAll = 0
    totalAll = 0
    with torch.no_grad():
        for i in range(5):
            for batch, (inputs, targets) in enumerate(testloader):
                pred = np.zeros(shape=(targets.shape[0], 10))
                label = targets.numpy()
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                for i in range(targets.shape[0] // 100):
                    outputs = model(inputs[i * 100:(i + 1) * 100, :, :, :])
                    results = outputs.data.cpu().numpy()
                    index = np.zeros(shape=(results.shape[0], 10))
                    aaa = np.argmax(results, axis=1)
                    for j in range(aaa.shape[0]):
                        index[j, aaa[j]] = 1
                    pred[i * 100:(i + 1) * 100, :] += index
                    # loss = criterion(outputs, targets) + mcloss(feature, targets) * 10
                    # loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)

        pred = np.argmax(pred, axis=1)
        import sklearn.metrics as metrics
        accuracy = metrics.accuracy_score(pred, label) * 100
        error = 100 - accuracy
        print('using:    batch=%04d' % batch, '  accuracy=%.2f' % accuracy, end='\r')
    print('USING:    batch=%04d' % batch, '  accuracy=%.2f' % accuracy)
    f = open(log_txt, 'a+')
    line = str(batch) + "  " + str((correctAll / totalAll * 100.0)) + '\n'
    f.write(line)
    f.close()

    print('')
    return correctAll / totalAll * 100.0


def test(use_cuda, testloader):
    model.eval()
    correctAll = 0
    totalAll = 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            pred = np.zeros(shape=(targets.shape[0], 10))
            label = targets.numpy()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            for i in range(targets.shape[0] // 100):
                outputs = model(inputs[i * 100:(i + 1) * 100, :, :, :])
                pred[i * 100:(i + 1) * 100, :] = outputs.data.cpu()
                # loss = criterion(outputs, targets) + mcloss(feature, targets) * 10
                # loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
                correct, total = accuracy(outputs.data, targets[i * 100:(i + 1) * 100].data)
                correctAll += correct
                totalAll += total
                print('using:    batch=%04d' % batch, '  accuracy=%.2f' % (correct / total * 100.0), end='\r')
        print('USING:    batch=%04d' % batch, '  accuracy=%.2f' % (correctAll / totalAll * 100.0))
    f = open(log_txt, 'a+')
    line = str(batch) + ' ' + str((correctAll / totalAll * 100.0)) + '\n'
    f.write(line)
    f.close()

    # print('')
    # from cleanlab.pruning import get_noise_indices
    # ordered_label_errors = get_noise_indices(
    #     s=label,
    #     psx=pred,
    #     sorted_index_method='normalized_margin',  # Orders label errors
    #     n_jobs=1
    # )

    return correctAll / totalAll * 100.0


def del_mis_label(use_cuda, testloader):
    model.eval()
    correctAll = 0
    totalAll = 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            pred = np.zeros(shape=(targets.shape[0], 10))
            label = targets.numpy()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            for i in range(targets.shape[0] // 100):
                outputs = model(inputs[i * 100:(i + 1) * 100, :, :, :])
                pred[i * 100:(i + 1) * 100, :] = outputs.data.cpu()
                # loss = criterion(outputs, targets) + mcloss(feature, targets) * 10
                # loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
                correct, total = accuracy(outputs.data, targets[i * 100:(i + 1) * 100].data)
                correctAll += correct
                totalAll += total
                print('using:    batch=%04d' % batch, '  accuracy=%.2f' % (correct / total * 100.0), end='\r')
        print('USING:    batch=%04d' % batch, '  accuracy=%.2f' % (correctAll / totalAll * 100.0))
    f = open(log_txt, 'a+')
    line = str(batch) + "  " + str((correctAll / totalAll * 100.0)) + '\n'
    f.write(line)
    f.close()

    print('')
    from cleanlab.pruning import get_noise_indices
    ordered_label_errors = get_noise_indices(
        s=label,
        psx=pred,
        sorted_index_method='normalized_margin',  # Orders label errors
        n_jobs=1
    )

    return correctAll / totalAll * 100.0, ordered_label_errors


def addon_test_sample(use_cuda, testloader):
    model.eval()
    correctAll = 0
    totalAll = 0

    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            pred = np.zeros(shape=(targets.shape[0], 10))
            label = targets.numpy()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            for i in range(targets.shape[0] // 100):
                outputs = model(inputs[i * 100:(i + 1) * 100, :, :, :])
                pred[i * 100:(i + 1) * 100, :] = F.softmax(outputs, dim=1).data.cpu()
                # loss = criterion(outputs, targets) + mcloss(feature, targets) * 10
                # loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
                correct, total = accuracy(outputs.data, targets[i * 100:(i + 1) * 100].data)
                correctAll += correct
                totalAll += total
                print('using:    batch=%04d' % batch, 'lr = %.6f', '  accuracy=%.2f' % (correct / total * 100.0),
                      end='\r')
        print('USING:    batch=%04d' % batch, '  accuracy=%.2f' % (correctAll / totalAll * 100.0))
    f = open(log_txt, 'a+')
    line = str(batch) + "  " + str((correctAll / totalAll * 100.0)) + '\n'
    f.write(line)
    f.close()

    test_idx = []
    pred_test_label = []
    pred_label = np.argmax(pred, axis=1)
    for i in range(len(pred)):
        if pred[i, pred_label[i]] > 0.8:
            test_idx.append(i)
            pred_test_label.append(pred_label[i])

    return correctAll / totalAll * 100.0, test_idx, pred_test_label


import torchvision.datasets as datasets


class LocalFashionMNIST(datasets.MNIST):  # John: good idea, use local Fashion, do not download
    urls = []


class RandomErasing1(object):
    def __init__(self, probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import *

transform_train = transforms.Compose([
    # transforms.Resize(56),
    # transforms.Pad((14,14)),
    # transforms.Resize(56),
    gray2rgb(),
    RandAugment(),
    transforms.RandomCrop(28, padding=4),
    # transforms.RandomResizedCrop(28),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),

    transforms.ToTensor(),
    # transforms.Normalize((0.28604059698879547,0.28604059698879547,0.28604059698879547, ),
    #                      (0.3202489254311618,0.3202489254311618,0.3202489254311618,)),
    transforms.Normalize((0.1307, 0.1307, 0.1307,), (0.3081, 0.3081, 0.3081,)),
    # RandomErasing1(probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914, 0.4914, 0.4914]),
    # transforms.Normalize((0.1307,), (0.3081,)),
    # #mean=[0.485, 0.456, 0.406],
    #  std=[0.229, 0.224, 0.225]
    RandomErasing(),

])
# transform_train.transforms.insert(0, RandAugment(0,30))

# transform_val = transforms.Compose([
#     # transforms.Resize(56),
#     # transforms.Pad((14,14)),
#     # transforms.Resize(56),
#     gray2rgb(),
#     # RandAugment(),
#     transforms.RandomCrop(28, padding=4),
#     # transforms.RandomResizedCrop(28),
#     transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),

#     transforms.ToTensor(),
#     # transforms.Normalize((0.28604059698879547,0.28604059698879547,0.28604059698879547, ),
#     #                      (0.3202489254311618,0.3202489254311618,0.3202489254311618,)),
#     transforms.Normalize((0.1307,0.1307,0.1307,), (0.3081,0.3081,0.3081,)),
#     # RandomErasing1(probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914, 0.4914, 0.4914]),
#     # transforms.Normalize((0.1307,), (0.3081,)),
#     # #mean=[0.485, 0.456, 0.406],
#     #  std=[0.229, 0.224, 0.225]
#     RandomErasing(),

# ])

transform_test = transforms.Compose([
    # transforms.Resize(56),
    # transforms.Pad((14, 14)),
    # transforms.Resize(56),
    gray2rgb(),

    transforms.ToTensor(),
    # transforms.Normalize((0.28604059698879547,0.28604059698879547,0.28604059698879547,),
    #                      (0.3202489254311618,0.3202489254311618,0.3202489254311618,)),
    transforms.Normalize((0.1307, 0.1307, 0.1307,), (0.3081, 0.3081, 0.3081,)),
])
# John: use LocalFashionMNIST to avoid download fashion behind proxy.  1)mkdir ./fashion/raw  2)download data and unzip to raw  3)mkdir ./fashion/processed  4)program tidy it
trainloader = data.DataLoader(
    datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform_train), pin_memory=True,
    batch_size=256, shuffle=True, num_workers=4)
testloader = data.DataLoader(
    datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform_test), pin_memory=True,
    batch_size=10000, shuffle=False, num_workers=4)
# valloader = data.DataLoader(
#     datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform_test), pin_memory=True,
#     batch_size=10000, shuffle=False, num_workers=4)

import numpy as np


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # mis_label = [1,2,3,4,5]
    # from my_dataloader import fashionmnist_addon
    # trainloader = data.DataLoader(
    #     fashionmnist_addon(root=data_path, train=True, download=True, transform=transform_train,idx_list=mis_label), pin_memory=True,
    #     batch_size=128, shuffle=True, num_workers=4)

    # a = np.asarray([1,2,3,4])
    # b = np.asarray([2,2,2,2,])
    # index = np.arange(a.shape[0])
    # aaa = index[a==b]
    import torch
    import os

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # torch.cuda.current_device()
    seed_torch(seed=42)

    use_cuda = torch.cuda.is_available()
    model = TPJ(num_classes=10, depth=40, widen_factor=4)  # John: hyper-parameter:  widen_factor=1 or 4

    if use_cuda:
        model = model.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing()

    from lightning import FMix

    fmix_obj = FMix(size=(28, 28))

    learning_rate_init = 0.1

    from SGD import SGDW_GCC

    optimizer = SGDW_GCC(model.parameters(), lr=learning_rate_init, momentum=0.9,
                         weight_decay=0.0002)  # use SGD firstly, not Adam
    # from torch_lr_finder import LRFinder

    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(trainloader, val_loader=testloader, end_lr=1, num_iter=100, step_mode="linear")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()
    # from lookahead import Lookahead
    #
    # optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    swa_train(use_cuda, trainloader, testloader, learning_rate_init)