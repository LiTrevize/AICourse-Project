import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers.bifpn import BIFPN
from data import voc_300, voc_512, voc_bifpn
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, extension):
        super(SSD, self).__init__()  # 调用父类nn.Module的构造函数
        self.phase = phase
        self.num_classes = num_classes

        # 如果传递到__init__()函数的参数num_classes=21,那么方括号内值为1，选择voc数据集，否则选择coco数据集
        if 'bifpn' in extension:
            self.cfg = eval(('coco', 'voc')[num_classes == 21] + '_bifpn')
        else:
            self.cfg = eval(('coco', 'voc')[num_classes == 21] + '_' + ('300', '512')[size == 512])

        # prior 是大小为[8732,4]的tensor，记录所有8732个先验框的x,y,h,w
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size
        self.extension = extension

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        if 'bifpn' in self.extension:
            self.neck = BIFPN(in_channels=[512, 1024, 512, 256, 256],
                              out_channels=256,
                              stack=2,
                              num_outs=5)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if 'iou_loss' in self.extension:
            self.iou = nn.ModuleList(head[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # Detect函数保存在layers/functions里
            # 参数分别为：num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45, self.cfg['variance'])

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch,num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors,4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        if 'iou_loss' in self.extension:
            iou = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # normalize before first feature map for detection
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        feature_out = sources
        if 'bifpn' in self.extension:
            feature_out = self.neck(sources)

        # print("shape of feature_out: ")
        # for i in feature_out:
        #     print(i.shape)

        # sources是含有5个Tensor的元组，每个Tensor都是一个batch的图片经过网络后某一特征图的特征
        # sources[0]: torch.Size([32, 512, 64, 64])
        # sources[1]: torch.Size([32, 1024, 32, 32])
        # sources[2]: torch.Size([32, 512, 16, 16])
        # sources[3]: torch.Size([32, 256, 8, 8])
        # sources[4]: torch.Size([32, 256, 4, 4])

        # apply multibox head to source layers
        if 'iou_loss' in self.extension:
            for (x, l, c, i) in zip(feature_out, self.loc, self.conf, self.iou):
                # permute函数将所得到的特征图tensor的通道维度放到最后一个维度
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
                iou.append(i(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(feature_out, self.loc, self.conf):
                # permute函数将所得到的特征图tensor的通道维度放到最后一个维度
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 在cat操作之前，loc是一个包含5个Tensor的元组，每个元组代表
        # 一个特征图经过头网络后关于每个点每个先验框偏移的4个预测
        # loc[0]: torch.Size([32, 64, 64, 16])
        # loc[1]: torch.Size([32, 32, 32, 24])
        # loc[2]: torch.Size([32, 16, 16, 24])
        # loc[3]: torch.Size([32, 8, 8, 24])
        # loc[4]: torch.Size([32, 4, 4, 16])
        # 执行cat操作后，loc为：torch.Size([32, 98048])
        # conf 和 loc类似
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if 'iou_loss' in self.extension:
            iou = torch.cat([o.view(o.size(0), -1) for o in iou], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            # print(loc.view(loc.size(0), -1, 4).shape)
            if 'iou_loss' in self.extension:
                output = (
                    loc.view(loc.size(0), -1, 4),                   # torch.Size([32, 24512, 4])
                    conf.view(conf.size(0), -1, self.num_classes),  # torch.Size([32, 24512, 21])
                    iou.view(iou.size(0),-1),                       # torch.Size([batch_size, 24512])
                    self.priors                                     # torch.Size([24512, 4])
                )
            else:
                output = (
                    loc.view(loc.size(0), -1, 4),  # torch.Size([32, 8732, 4])
                    conf.view(conf.size(0), -1, self.num_classes),  # torch.Size([32, 8732, 21])
                    self.priors  # torch.Size([8732, 4])
                )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# SSD 使用了VGG16(D)卷积部分（5层卷积，一般标记为Conv5）作为基础网络，
# 后面加了 1024 × 3 × 3、1024 × 1 × 1 两个卷积层，这两个卷积层后都有 RELU 层
# Conv2d-4_3、Conv2d-7_1是要用来做特征提取的层
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# 辅助特征层, 搭建在之前的vgg之后，之前VGG网络的输出形状最终为1024*19*19
# 新添加的8个卷积层记为Conv2d-1_1，Conv2d-2_1，Conv2d-3_1，Conv2d-4_1，Conv2d-5_1，Conv2d-6_1，Conv2d-7_1，Conv2d-8_1
# 其中（Conv2d-2_1、Conv2d-4_1、Conv2d-6_1、Conv2d-8_1）用于多尺度特征提取
# ‘S’用来判断是否3*3的卷积步长是2，padding是1
def add_extras(cfg, i, batch_norm=False, ssd512=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    print("..................................................")
    print("shape of layers: ")
    for i in layers:
        print(i)
    if ssd512:  # 对于SSD512额外添加的层
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, bifpn=False, iou_loss=False):
    loc_layers = []
    conf_layers = []
    iou_layers = []

    vgg_source = [21, -2]
    # 对vgg网络结构中的Conv2d-4_3、Conv2d-7_1层通过卷积提取特征
    for k, v in enumerate(vgg_source):
        if bifpn:
            if iou_loss==False:
                loc_layers += [nn.Conv2d(256,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(256,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
            else:
                loc_layers += [nn.Conv2d(256,
                                         cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(256,
                                          cfg[k] * num_classes, kernel_size=3, padding=1)]
                iou_layers += [nn.Conv2d(256,
                                          cfg[k], kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 对extra_layers中的（Conv2d-2_1、Conv2d-4_1、Conv2d-6_1、Conv2d-8_1）层通过卷积提取特征
    for k, v in enumerate(extra_layers[1::2], 2):
        if bifpn:
            if iou_loss==False:
                loc_layers += [nn.Conv2d(256,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(256,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
            else:
                loc_layers += [nn.Conv2d(256,
                                         cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(256,
                                          cfg[k] * num_classes, kernel_size=3, padding=1)]
                iou_layers += [nn.Conv2d(256,
                                          cfg[k], kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(v.out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
    if iou_loss:
        return vgg, extra_layers, (loc_layers, conf_layers, iou_layers)
    else:
        return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    'bifpn': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4, 4],
    'bifpn': [4, 6, 6, 6, 4],
}


def build_ssd(phase, size=300, num_classes=21, extension=[]):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300 or size!=512:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    # head_ 的形式为([],[]),前后两个列表中各有6个卷积层
    # 前一个列表中每一个卷积层负责代表预测6个feature map中的某一个上任意一点的k个先验框的4个维度的偏差，
    # 比如第一个卷积层是：Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 后一个列表中每一个卷积层负责代表预测6个feature map中的某一个上任意一点的k个先验框的21个类别的置信度，
    # 比如第一个卷积层是：Conv2d(512, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if 'bifpn' in extension:
        if 'iou_loss' in extension:
            base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                             add_extras(extras['bifpn'], 1024, ssd512=False),
                                             mbox['bifpn'], num_classes, True,True)
        else:
            base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                         add_extras(extras['bifpn'], 1024, ssd512=False),
                                         mbox['bifpn'], num_classes, True,False)
    else:
        base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                         add_extras(extras[str(size)], 1024, ssd512=(size == 512)),
                                         mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes, extension)
