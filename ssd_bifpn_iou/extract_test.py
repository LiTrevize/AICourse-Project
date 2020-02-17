import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pickle
from scipy import interpolate

parser = argparse.ArgumentParser(
    description='Plot training loss diagrams')
# specify the model
parser.add_argument('--model',
                    choices=['ssd_bifpn', 'ssd_bifpn_sr', 'ssd_bifpn_iou_loss',
                             'ssd300', 'ssd512'],
                    type=str, help='model type')
parser.add_argument('--eval_dir', default='eval/',
                    type=str, help='directory to the weights')
parser.add_argument('--plot', action='store_true',
                    help='directory to the output')
parser.add_argument('--show', action='store_true',
                    help='show on display')
parser.add_argument('--output_dir', default='eval/',
                    type=str, help='directory to the output')
parser.add_argument('--f', action='store_true',
                    help='force to retrieve data from weight files')
args = parser.parse_args()

MODELS = ['ssd300', 'ssd_bifpn', 'ssd_bifpn_iou_loss']
size_labels = ['XS', 'S', 'M', 'L', 'XL', 'Total']

prec_dict = {}
ap_dict = {}

for model in MODELS:
    rec = np.zeros((6, 101))
    prec = np.zeros((6, 101))
    ap = np.zeros(6)
    counts = 0
    filedir = args.eval_dir + model + '/test/'
    for filename in os.listdir(filedir):
        if 'detection' in filename:
            continue
        with open(filedir + filename, 'rb') as f:
            metric = pickle.load(f)
        # print(np.array(metric['ap']).shape,np.array(metric['rec']).shape,np.array(metric['prec']).shape)
        # print(metric['prec'][-1])
        for i in range(6):
            x = np.array(metric['rec'][i])
            y = np.array(metric['prec'][i])
            z = x[1:] - x[:len(x) - 1]
            z = (z != 0)
            z = np.append(z, True)
            # print(x[z])
            try:
                f = interpolate.interp1d(x[z], y[z], kind='quadratic')
                # rec[i] +=
                newx = np.linspace(0, 1, 101)
                l = sum(newx < x[0])
                h = sum(newx <= x[-1])
                prec[i][l:h] = prec[i][l:h] + f(newx[l:h])
                # print(np.array(metric['rec'][i]).shape)
            except:
                print('invalid')
            ap[i] += metric['ap'][i]
        counts += 1
    # rec /= counts
    prec /= counts
    ap /= counts

    prec_dict[model] = prec
    ap_dict[model] = ap
    print(model)
    for i in range(5):
        print('\t{}: {:.4f}'.format(size_labels[i], ap[i]))
        # print('\trecall: {:.4f}\t prec: {:.4f}\t AP: {:.4f}'.format(rec[i], prec[i], ap[i]))
    print('\tTotal: {:.4f}'.format(ap[-1]))
    # print('\trecall: {:.4f}\t prec: {:.4f}\t AP: {:.4f}'.format(rec[-1], prec[-1], ap[-1]))

    if args.plot:
        plt.figure()
        for i in range(5):
            plt.plot(np.linspace(0, 1, 101), prec[i], label=size_labels[i])
        plt.plot(np.linspace(0, 1, 101), prec[i], label='Total')
        plt.legend()
        plt.title('Precision-Recall for ' + model)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(args.output_dir + model + '_pr' + '.png')
        if args.show:
            plt.show()

if args.plot:
    for i in range(6):
        plt.figure()
        for m in MODELS:
            plt.plot(np.linspace(0, 1, 101), prec_dict[m][i], label=m)
            # label=m + ', ' + 'mAP={:.4f}'.format(ap_dict[m][i]))
        plt.legend()
        plt.title('Precision-Recall for ' + size_labels[i])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(args.output_dir + size_labels[i] + '_pr' + '.png')
        if args.show:
            plt.show()
