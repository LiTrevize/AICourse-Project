cd ssd_bifpn

# train ssd_bifpn
python train.py --model ssd_bifpn

# train ssd_bifpn with super resolution
python train.py --model ssd_bifpn -sr --batch_size 24
CUDA_VISIBLE_DEVICES=0 python train.py --model ssd_bifpn -sr --resume weights/ssd_bifpn_VOC_sr_138000_2.2238.pth --start_iter 138000 --batch_size 24

# train ssd300 or ssd512
python train.py --model ssd300

# plot training loss
python vis_loss.py --model ssd_bifpn_sr --plot

# evaluate
CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model weights/VOC_bifpn.pth --model ssd_bifpn

CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model weights/ssd_bifpn_VOC_sr.pth --model ssd_bifpn -sr

CUDA_VISIBLE_DEVICES=0 python eval.py --model ssd_bifpn_iou_loss --trained_model weights/ssd_bifpn_iou_loss_VOC_iou_loss124000_1.2295_1.3632.pth

CUDA_VISIBLE_DEVICES=0 python eval.py --trained_model weights/VOC300.pth --model ssd300
