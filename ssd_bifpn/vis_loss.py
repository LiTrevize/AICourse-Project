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
parser.add_argument('-f', action='store_true',
                    help='force to retrieve data from weight files')
args = parser.parse_args()

# keywords in args.model
kws = args.model.split('_')
# maps of iteration to loss
loss_map = {}


def select_model(x):
    for kw in kws:
        if kw not in x:
            return False
    return True


def get_train_loss_from_ckp():
    files = os.listdir(args.weights_dir)
    for w in filter(select_model, files):
        w = w[:len(w) - 4]
        ws = w.split('_')
        l = float(ws[-1])
        iter = int(ws[-2])
        loss_map[iter] = l


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
        train_loss = {'iter': iter, 'loss': loss}
        with open(output_name + '.pkl', 'wb') as f:
            pickle.dump(train_loss, f)

    if args.plot:
        plt.plot(train_loss['iter'], train_loss['loss'])
        plt.title('training loss of ' + args.model)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(args.output_dir + args.model + '.png')
        plt.show()
