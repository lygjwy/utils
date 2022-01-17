import argparse
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})

from adjustText import adjust_text

import torch
import torch.nn.functional as F

from metrics import compute_all_metrics


def draw_line(data, shape, label):
    x = list(range(len(data)))
    max_idx = data.index(max(data))
    min_idx = data.index(min(data))
    last_idx = len(data) - 1

    l1 = plt.plot(x, data, shape, label=label)
    texts = []
    plt.plot(max_idx, data[max_idx], 'bs')
    max_show_text = '('+str(max_idx)+','+str(data[max_idx])+')'
    texts.append(plt.text(max_idx, data[max_idx], max_show_text))
    
    plt.plot(min_idx, data[min_idx], 'ks')
    min_show_text = '('+str(min_idx)+','+str(data[min_idx])+')'
    texts.append(plt.text(min_idx, data[min_idx], min_show_text))

    plt.plot(last_idx, data[last_idx], 'ms')
    last_show_text = '('+str(last_idx)+','+str(data[last_idx])+')'
    texts.append(plt.text(last_idx, data[last_idx], last_show_text))
    adjust_text(texts, x=x, y=data, autoalign='y',
            only_move={'points':'xy', 'text':'xy'}, force_points=1,
            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

def single_visualize(data, label_data, label_x, label_y, fig_path):
    plt.clf()
    draw_line(data, 'c--', label_data)

    # plt.title(label)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


def double_visualize(data_a, data_b, label_a, label_b, label_x, label_y, fig_path):
    # visualize the trend
    assert len(data_a) == len(data_b)
    plt.clf()
    draw_line(data_a, 'r--', label_a)
    draw_line(data_b, 'g--', label_b)

    # plt.title('Loss')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


def conf_from_logits(logits):
    conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0].numpy()
    return conf


def kl_conf_from_logits(logits):
    softmaxs = torch.softmax(logits, dim=1)
    uniform_dis = torch.ones_like(softmaxs) * (1 / softmaxs.shape[1])
    kl_conf = torch.sum(F.kl_div(softmaxs.log(), uniform_dis, reduction='none'), dim=1).numpy()
    return kl_conf


def eva_ood(exp_path, epochs, pattern, mode):
    fpr_at_tprs, aurocs, aupr_ins, aupr_outs = [], [], [], []
    for i in range(epochs):
        epoch_path = exp_path / str(i)
        if pattern in ['logits', 'rec_logits']:
            id_logits = np.load(str(epoch_path / ('-'.join([pattern, id]) + '.npy')))
            total_fpr_at_tpr, total_auroc, total_aupr_in, total_aupr_out = [], [], [], []
            for ood in oods:
                ood_logits = np.load(str(epoch_path / ('-'.join([pattern, ood]) + '.npy')))
                id_logits, ood_logits = torch.Tensor(id_logits), torch.Tensor(ood_logits)
                # calculate mean auroc & aupr_in & aupr_out & fpr_tpr
                if mode == 'normal':
                    id_scores = conf_from_logits(id_logits)
                    ood_scores = conf_from_logits(ood_logits)
                elif mode == 'kl':
                    id_scores = kl_conf_from_logits(id_logits)
                    ood_scores = kl_conf_from_logits(ood_logits)
                else:
                    raise RuntimeError('---> invalid mode: '.format(mode))
                confs = np.concatenate([id_scores, ood_scores])
                id_labels = np.zeros(id_scores.shape[0])
                ood_labels = np.ones(ood_scores.shape[0])
                labels = np.concatenate([id_labels, ood_labels])
                fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(confs, labels)
                total_fpr_at_tpr.append(fpr_at_tpr)
                total_auroc.append(auroc)
                total_aupr_in.append(aupr_in)
                total_aupr_out.append(aupr_out)
            
            fpr_at_tprs.append(np.mean(total_fpr_at_tpr))
            aurocs.append(np.mean(total_auroc))
            aupr_ins.append(np.mean(total_aupr_in))
            aupr_outs.append(np.mean(total_aupr_out))
        elif pattern == 'combined':
            id_logits = np.load(str(epoch_path / ('-'.join(['logits', id]) + '.npy')))
            id_rec_logits = np.load(str(epoch_path / ('-'.join(['rec_logits', id]) + '.npy')))
            total_fpr_at_tpr, total_auroc, total_aupr_in, total_aupr_out = [], [], [], []
            for ood in oods:
                ood_logits = np.load(str(epoch_path / ('-'.join(['logits', ood]) + '.npy')))
                ood_rec_logits = np.load(str(epoch_path / ('-'.join(['rec_logits', ood]) + '.npy')))
                id_logits, id_rec_logits, ood_logits, ood_rec_logits = torch.Tensor(id_logits), torch.Tensor(id_rec_logits), torch.Tensor(ood_logits), torch.Tensor(ood_rec_logits)
                # calculate mean auroc & aupr_in & aupr_out & fpr_tpr
                if mode == 'normal':
                    id_scores = conf_from_logits(id_logits) + conf_from_logits(id_rec_logits)
                    ood_scores = conf_from_logits(ood_logits) + conf_from_logits(ood_rec_logits)
                elif mode == 'kl':
                    id_scores = kl_conf_from_logits(id_logits) + kl_conf_from_logits(id_rec_logits)
                    ood_scores = kl_conf_from_logits(ood_logits) + kl_conf_from_logits(ood_rec_logits)
                else:
                    raise RuntimeError('---> invalid mode: '.format(mode))
                confs = np.concatenate([id_scores, ood_scores])
                id_labels = np.zeros(id_scores.shape[0])
                ood_labels = np.ones(ood_scores.shape[0])
                labels = np.concatenate([id_labels, ood_labels])
                fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(confs, labels)
                total_fpr_at_tpr.append(fpr_at_tpr)
                total_auroc.append(auroc)
                total_aupr_in.append(aupr_in)
                total_aupr_out.append(aupr_out)
            
            fpr_at_tprs.append(np.mean(total_fpr_at_tpr))
            aurocs.append(np.mean(total_auroc))
            aupr_ins.append(np.mean(total_aupr_in))
            aupr_outs.append(np.mean(total_aupr_out))

        else:
            raise RuntimeError('---> invalid pattern: {}'.format(pattern))
    # plot each
    # single_visualize(fpr_at_tprs, 'fpr@95tpr', 'epoch', 'fpr@95tpr', str(exp_path / '-'.join([pattern, mode, 'fpr_at_tprs.png'])))
    single_visualize(aurocs, 'auroc', 'epoch', 'auroc', str(exp_path / '-'.join([pattern, mode, 'aurocs.png'])))
    # single_visualize(aupr_ins, 'aupr_in', 'epoch', 'aupr_in', str(exp_path / '-'.join([pattern, mode, 'aupr_ins.png'])))
    # single_visualize(aupr_outs, 'aupr_out', 'epoch', 'aupr_out', str(exp_path / '-'.join([pattern, mode, 'aupr_outs.png'])))


parser = argparse.ArgumentParser('analyse result')
parser.add_argument('--root', type=str, default='/mnt/e/Exp/', help='exp dir')
# parser.add_argument('--paradigms', nargs='*', default=['classify'], help='training paradigm')
# parser.add_argument('--lambda_recs', nargs='*', default=[], help='lambda_recs')

args = parser.parse_args()

id = 'cifar10'
oods = ['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun']
root_path = Path(args.root)

# paradigms = args.paradigms
# lambda_recs = args.lambda_recs


print('---> start.')
# for paradigm in paradigms:

#     if paradigm == 'classify':
#         exp_path = root_path / paradigm
#         print('---> parse exp: {}'.format(paradigm))
#         console_path = exp_path / 'console.log'

#         with open(console_path) as f:
#             contents = f.readlines()
#         cla_accs, rec_errs = [], []

#         for line in contents:
#             if line.startswith('[cla_loss:'):
#                 _, _, _, cla_acc = re.findall(r'\d+\.+\d*', line) 
#                 cla_accs.append(float(cla_acc))
#         train_cla_accs, test_cla_accs = cla_accs[::2], cla_accs[1::2]
#         # plot cla_acc-epoch.png & rec_err-epoch.png
#         double_visualize(train_cla_accs, test_cla_accs, 'train-acc', 'test-acc', 'epoch', 'accuracy', str(exp_path / 'cla_accs.png'))
        
#         eva_ood(exp_path, 200, 'logits', 'normal')
#         eva_ood(exp_path, 200, 'logits', 'kl')

#     elif paradigm in ['joint']:
#         for lambda_rec in lambda_recs:
#             exp_path = root_path / '-'.join([paradigm, lambda_rec])
#             print('---> parse exp: {} {}'.format(paradigm, lambda_rec))
#             console_path = exp_path / 'console.log'
            
#             with open(console_path) as f:
#                 contents = f.readlines()
#             cla_accs, rec_errs = [], []

#             for line in contents:
#                 if line.startswith('[cla_loss:'):
#                     _, rec_err, _, cla_acc = re.findall(r'\d+\.+\d*', line) 
#                     cla_accs.append(float(cla_acc))
#                     rec_errs.append(float(rec_err))
#             train_cla_accs, test_cla_accs = cla_accs[::2], cla_accs[1::2]
#             train_rec_errs, test_rec_errs = rec_errs[::2], rec_errs[1::2]
#             # plot cla_acc-epoch.png & rec_err-epoch.png
#             double_visualize(train_cla_accs, test_cla_accs, 'train-acc', 'test-acc', 'epoch', 'accuracy', str(exp_path / 'cla_accs.png'))
#             double_visualize(train_rec_errs, test_rec_errs, 'train-rec', 'test-rec', 'epoch', 'rec_err', str(exp_path / 'rec_errs.png'))
            
#             for pattern in ['logits', 'rec_logits', 'combined']:
#                 for mode in ['normal', 'kl']:
#                     eva_ood(exp_path, 200, pattern, mode)

#     else:
#         raise RuntimeError('---> invalid paradigm: {}'.format(paradigm))

print('>>> Parse exp: {}'.format(str(root_path)))
console_path = root_path / 'console.log'
with open(console_path) as f:
    contents = f.readlines()
train_rec_errs, test_rec_errs = [], []

for line in contents:
    if line.startswith('[rec loss:'):
        train_rec_errs.append(float(re.findall(r'\d+\.+\d*', line)[0]))
    if line.startswith('[rec_loss:'):
        test_rec_errs.append(float(re.findall(r'\d+\.+\d*', line)[0]))
fig_path = root_path / 'rec_errs_400.png'
double_visualize(train_rec_errs[500:], test_rec_errs[500:], 'train rec err', 'test rec err', 'epoch', 'rec err', str(fig_path))

print('---> done.')


            