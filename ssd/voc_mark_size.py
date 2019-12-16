from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    # parser.add_argument('--trained_model',
    #                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
    #                    help='Trained state_dict file path to open')
    # parser.add_argument('--trained_model',
    #                     default='weights/VOC.pth', type=str,
    #                     help='Trained state_dict file path to open')
    # parser.add_argument('--save_folder', default='eval/', type=str,
    #                     help='File path to save results')
    # parser.add_argument('--confidence_threshold', default=0.01, type=float,
    #                     help='Detection confidence threshold')
    # parser.add_argument('--top_k', default=5, type=int,
    #                     help='Further restrict the number of predictions to parse')
    # parser.add_argument('--cuda', default=True, type=str2bool,
    #                     help='Use cuda to train model')
    parser.add_argument('--voc_root', default=VOC_ROOT,
                        help='Location of VOC root directory')
    # parser.add_argument('--cleanup', default=True, type=str2bool,
    #                     help='Cleanup and remove results files following eval')

    return parser.parse_args()


args = parse_args()
annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'

if __name__ == '__main__':
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),  # resize to 300*300 and - mean
                           VOCAnnotationTransform(keep_difficult=True))

    # cls * id/size
    num_cls = len(labelmap)
    dets_id = [[] for i in range(num_cls)]
    dets_size = [[] for i in range(num_cls)]
    print('pulling annotation...')
    for i in range(len(dataset)):
        imgid, gt = dataset.pull_anno(i)
        for j, det in enumerate(gt):
            det_size = (det[2] - det[0]) * (det[3] - det[1])
            cls = det[-1]
            dets_id[cls].append(imgid)
            dets_size[cls].append(det_size)
        if i % 100 == 0:
            print('\rProgress: {}%'.format(int(100.0 * i / len(dataset))), end='')
    print()

    # extra-small (XS: bottom 10%)
    # small (S: next 20%)
    # medium (M: next 40%)
    # large (L: next 20%)
    # extra-large (XL: next 10%)
    sizes = ['XS', 'S', 'M', 'L', 'XL']
    sizemap = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4}

    sorted_size = [sorted(dets_size[i]) for i in range(num_cls)]

    # determine the boundary and replace actual size with label
    for i in range(len(sorted_size)):
        cls_size = sorted_size[i]
        num = len(cls_size)
        XS = cls_size[int(num * 0.1)]
        S = cls_size[int(num * 0.3)]
        M = cls_size[int(num * 0.7)]
        L = cls_size[int(num * 0.9)]
        print(i, XS, S, M, L)

        #
        for j in range(num):
            if dets_size[i][j] <= XS:
                dets_size[i][j] = 0
            elif dets_size[i][j] <= S:
                dets_size[i][j] = 1
            elif dets_size[i][j] <= M:
                dets_size[i][j] = 2
            elif dets_size[i][j] <= L:
                dets_size[i][j] = 3
            else:
                dets_size[i][j] = 4

    # cls_imgs: (cls, imgid, list of detection size)
    print('rearranging data format...')
    cls_imgs = [{} for i in range(num_cls)]
    for cls in range(num_cls):
        for _, imgid in dataset.ids:
            for i in range(len(dets_id[cls])):
                if dets_id[cls][i] == imgid:
                    if imgid not in cls_imgs[cls].keys():
                        cls_imgs[cls][imgid] = [dets_size[cls][i]]
                    else:
                        cls_imgs[cls][imgid].append(dets_size[cls][i])

    # for i in range(len(cls_imgs)):
    #     print(cls_imgs[i].keys())
    #     print(cls_imgs[i])
    with open("det_size.pkl", 'wb') as f:
        pickle.dump(cls_imgs, f, pickle.HIGHEST_PROTOCOL)
