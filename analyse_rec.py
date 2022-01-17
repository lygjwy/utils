import argparse
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

from analyse_result import double_visualize
plt.rcParams.update({'font.size': 6})

from adjustText import adjust_text

import torch
import torch.nn.functional as F

from metrics import compute_all_metrics


parser = argparse.ArgumentParser('analyse reconstruction')
parser.add_argument('--root', type=str, default='/mnt/c/Users/jwy/Desktop', help='exp dir')
parser.add_argument('--paradigms', nargs='*', default=['reconstruct'], help='training paradigm')
parser.add_argument('--lambda_recs', nargs='*', default=[], help='lambda recs')

args = parser.parse_args()

id = 'cifar10'
oods = ['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun']
root_path = Path(args.root)

paradigms = args.paradigms
lambda_recs = args.lambda_recs


print('---> start.')
for paradigm in paradigms:
    if paradigm == 'reconstruct':
        exp_path = root_path / paradigm
        print('---> parse exp: {}'.format(paradigm))
        console_path = exp_path / 'console.log'

        with open(console_path) as f:
            contents = f.readlines()
        rec_errs = []
        for line in contents:
            if line.startswith('[cla_loss:'):
                _, rec_err, _, _ = re.findall(r'\d+\.+\d*', line)
                rec_errs.append(float(rec_err))
        train_rec_errs, test_rec_errs = rec_errs[::2], rec_errs[1::2]
        double_visualize(train_rec_errs, test_rec_errs, 'train-rec', 'test-rec', 'epoch', 'rec_err', str(exp_path / 'rec_errs.png'))

    elif paradigm == 'joint':
        for lambda_rec in lambda_recs:
            exp_path = root_path / '-'.join([paradigm, lambda_rec])
            print('---> parse exp: {} {}'.format(paradigm, lambda_rec))
            console_path = exp_path / 'console.log'

            with open(console_path) as f:
                contents = f.readlines()
            cla_accs, rec_errs = [], []

            for line in contents:
                if line.startswith('[cla_loss:'):
                    _, rec_err, _, cla_acc = re.findall(r'\d+\.+\d*', line) 
                    cla_accs.append(float(cla_acc))
                    rec_errs.append(float(rec_err))
            train_cla_accs, test_cla_accs = cla_accs[::2], cla_accs[1::2]
            train_rec_errs, test_rec_errs = rec_errs[::2], rec_errs[1::2]
            # plot cla_acc-epoch.png & rec_err-epoch.png
            double_visualize(train_cla_accs, test_cla_accs, 'train-acc', 'test-acc', 'epoch', 'accuracy', str(exp_path / 'cla_accs.png'))
            double_visualize(train_rec_errs, test_rec_errs, 'train-rec', 'test-rec', 'epoch', 'rec_err', str(exp_path / 'rec_errs.png'))

    else:
        raise RuntimeError('---> invalid paradigm: {}'.format(paradigm))

print('---> done')