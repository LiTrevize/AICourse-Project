import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pickle

parser = argparse.ArgumentParser(
    description='Plot training loss diagrams')
# specify the model
parser.add_argument('--model',
                    choices=['ssd_bifpn', 'ssd_bifpn_sr', 'ssd_bifpn_iou_loss',
                             'ssd300', 'ssd512'],
                    type=str, help='model type')
parser.add_argument('--weights_dir', default='weights/',
                    type=str, help='directory to the weights')
parser.add_argument('--plot', action='store_true',
                    help='directory to the output')
parser.add_argument('--output_dir', default='output/',
                    type=str, help='directory to the output')
parser.add_argument('--f', action='store_true',
                    help='force to retrieve data from weight files')
args = parser.parse_args()

# keywords in args.model

# maps of iteration to loss
loss_map = {}
ALL_KEYS = ['bifpn', 'sr', 'iou', '300', '512']
INCLUSIVE = [False for i in range(len(ALL_KEYS))]


def select_model(x):
    for i, k in enumerate(ALL_KEYS):
        if (k in x) != INCLUSIVE[i]:
            return False
    return True


def get_train_loss_from_ckp():
    for i, k in enumerate(ALL_KEYS):
        if k in args.model:
            INCLUSIVE[i] = True

    files = os.listdir(args.weights_dir)
    for w in filter(select_model, files):
        w = w[:len(w) - 4]
        ws = w.split('_')
        try:
            if args.model == 'ssd_bifpn_iou_loss':
                l_iou = float(ws[-1])
                l_lc = float(ws[-2])
                iter = int(ws[-3][4:])
                loss_map[iter] = (l_lc, l_iou)
            else:
                l = float(ws[-1])
                iter = int(ws[-2])
                loss_map[iter] = l
        except:
            print('invalid', w)


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_name = args.output_dir + args.model

    if os.path.exists(output_name + '.pkl') and not args.f:
        with open(output_name + '.pkl', 'rb') as f:
            train_loss = pickle.load(f)
    else:
        get_train_loss_from_ckp()
        iter = sorted(loss_map.keys())
        loss = [loss_map[k] for k in iter]
        train_loss = {'iter': np.array(iter), 'loss': np.array(loss)}
        with open(output_name + '.pkl', 'wb') as f:
            pickle.dump(train_loss, f)

    if args.plot:
        if args.model == 'ssd_bifpn_iou_loss':
            plt.plot(train_loss['iter'], train_loss['loss'][:, 0], label='loss_loc+loss_conf')
            plt.plot(train_loss['iter'], train_loss['loss'][:, 1], label='loss_iou')
            plt.plot(train_loss['iter'], np.sum(train_loss['loss'], axis=1), label='loss_total')
            plt.legend()
        else:
            plt.plot(train_loss['iter'], train_loss['loss'])
        plt.title('training loss of ' + args.model)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(args.output_dir + args.model + '.png')
        plt.show()
