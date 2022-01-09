import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

import torch

from metrics import compute_all_metrics


def draw_line(data, shape, label):
    x = list(range(len(data)))
    max_idx = data.index(max(data))
    min_idx = data.index(min(data))
    l1 = plt.plot(x, data, shape, label=label)
    plt.plot(max_idx, data[max_idx], 'bs')
    max_show_text = '('+str(max_idx)+','+str(data[max_idx])+')'
    plt.annotate(max_show_text, xy=(max_idx, data[max_idx]))
    min_show_text = '('+str(min_idx)+','+str(data[min_idx])+')'
    plt.annotate(min_show_text, xy=(min_idx, data[min_idx]))

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

id = 'cifar10'
oods = ['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun']
root_path = Path('/mnt/c/Users/jwy/Desktop/22.1.8/')

paradigms = ['no_pre', 'pre']
lambda_recs = ['0.00001', '0.00003', '0.00005', '0.00007', '0.00009', '0.0001', '0.0003', '0.0005']

# dirs = ['_'.join([paradigm, lambda_rec]) for paradigm in paradigms for lambda_rec in lambda_recs]

pattern = r'\d+\.+\d*'
print('---> start.')
for paradigm in paradigms:
    for lambda_rec in lambda_recs:
        exp_path = root_path / '-'.join([paradigm, lambda_rec])
        print('---> parse exp: {} {}'.format(paradigm, lambda_rec))
        console_path = exp_path / 'console.log'
        
        with open(console_path) as f:
            contents = f.readlines()
        cla_accs, rec_errs = [], []

        for line in contents:
            if line.startswith('[cla_loss:'):
                _, rec_err, _, cla_acc = re.findall(pattern, line) 
                cla_accs.append(float(cla_acc))
                rec_errs.append(float(rec_err))
        train_cla_accs, train_rec_errs = cla_accs[::2], cla_accs[1::2]
        test_cla_accs, test_rec_errs = rec_errs[::2], rec_errs[1::2]
        # plot cla_acc-epoch.png & rec_err-epoch.png
        double_visualize(train_cla_accs, test_cla_accs, 'train-acc', 'test-acc', 'epoch', 'accuracy', str(exp_path / 'cla_accs.png'))
        double_visualize(train_rec_errs, test_rec_errs, 'train-rec', 'test-rec', 'epoch', 'rec_err', str(exp_path / 'rec_errs.png'))
        
        fpr_at_tprs, aurocs, aupr_ins, aupr_outs = [], [], [], []
        # cal each epoch auroc, fpr_at_tpr, aupr_in, aupr_out
        for i in range(200):
            # traverse all epoch
            epoch_path = exp_path / str(i)
            # read all data set
            id_logits = np.load(str(epoch_path / ('-'.join(['logits', id]) + '.npy')))
            print(id_logits.shape)
            total_fpr_at_tpr, total_auroc, total_aupr_in, total_aupr_out = [], [], [], []
            for ood in oods:
                ood_logits = np.load(str(epoch_path / ('-'.join(['logits', ood]) + '.npy')))
                id_logits, ood_logits = torch.from_numpy(id_logits), torch.from_numpy(ood_logits)
                print(id_logits.shape)
                exit()
                # calculate mean auroc & aupr_in & aupr_out & fpr_tpr
                id_scores = torch.max(torch.softmax(id_logits, dim=1), dim=1)[0].numpy()
                id_labels = np.zeros(id_scores.shape[0])
                ood_scores = torch.max(torch.softmax(ood_logits, dim=1), dim=1)[0].numpy()
                ood_labels = np.ones(ood_scores.shape[0])
                confs = np.concatenate([id_scores, ood_scores])
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
        # plot each
        single_visualize(fpr_at_tprs, 'fpr@95tpr', 'epoch', 'fpr@95tpr', str(exp_path / 'fpr_at_tprs.png'))
        single_visualize(aurocs, 'auroc', 'epoch', 'auroc', str(exp_path / 'aurocs.png'))
        single_visualize(aupr_ins, 'aupr_in', 'epoch', 'aupr_in', str(exp_path / 'aupr_ins.png'))
        single_visualize(aupr_outs, 'aupr_out', 'epoch', 'aupr_out', str(exp_path / 'aupr_outs.png'))
print('---> done.')


            