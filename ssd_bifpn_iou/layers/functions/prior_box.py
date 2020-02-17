from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch

# 一次性返回一个大小为[8732,4]的tensor，即SSD网络不同尺度feature map上的所有先验框，每个先验框的四个元素分别为x,y,h,w
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                # +=意思是直接在mean现有列表的后面添加新的元素,mean保持是一个list，比如[1,2,3,4]+=[5,6,7,8]的结果是[1,2,3,4,5,6,7,8]
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)   # output.shape: torch.Size([8732, 4])
        # 如果clip参数为True，将output的tensor的每个元素压紧在[min,max]区间里
        # 比如原本第8732个先验框为[0.5,0.5,0.6223,1.2445]，压紧后变为[0.5,0.5,0.6223,1]，其实就相当于把超出区域的部分裁掉
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
