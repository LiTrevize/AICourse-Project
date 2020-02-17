# Improve SSD with Feature Fusion, IoU Loss and Super Resolution

In this project, we follow the one-stage detector design, and propose three methods to improve SSD for better small object detection performance:
+ BiFPN feature fusion
+ Additional IoU loss
+ Super-resolution input

## Acknowledgement
Our code is modified from amdegroot's ssd.pytorch, see the original repository [here](https://github.com/amdegroot/ssd.pytorch).

## Installation
+ Install PyTorch by selecting your environment on the official website and running the appropriate command. We currently support 
+ Clone this repository
    + We currently support Python 3.7 and PyTorch 1.3.1
+ Download PASCAL VOC2007 trainval & test and VOC2012 trainval
```
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
sh data/scripts/VOC2012.sh # <directory>
```
+ Note: Only VOC is tested, although you may also specify COCO.

## Evaluation
You can download our trained models from [jbox](https://jbox.sjtu.edu.cn/l/I510e7). By default we assume they are in `ssd_bifpn/weights` directory.

To evaluate a model, you need to specify the model type and weight file:
```
cd ssd_bifpn_iou
python eval.py --model [MODEL_NAME] --trained_model [WEIGHT_FILE] 
```
Currently, we support four types of model: `ssd300`, `ssd500`, `ssd_bifpn` and `ssd_bifpn_iou_loss`.

## Training
+ First download the fc-reduced VGG-16 PyTorch base network weights at the same link above as our trained models
+ You should also specify the model name and other parameters:
```
python train.py --model [MODEL_NAME]
```

+ You can also stop training and resume at the latest checkpoint:
```
python train.py --model [MODEL_NAME] --resume [CHECKPOINT_PATH] --start_iter [CHECKPOINT_ITER]
```

## Demo
You can run the demo by
```
python demo.py
```
Results are saved to `result_demo` by default.