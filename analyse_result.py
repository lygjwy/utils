from pathlib import Path
import re
import matplotlib.pyplot as plt

def visualize_ood(metrics, label, fig_path):
    x = list(range(len(metrics)))
    l = plt.plot(x, metrics, 'g--', label=label)

    # plt.title(label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


def visualize_loss(train_loss, test_loss, label, fig_path):
    # visualize the trend
    x = list(range(len(train_loss)))
    l1 = plt.plot(x, train_loss, 'r--', label='train loss')
    l2 = plt.plot(x, test_loss, 'g--', label='test loss')

    # plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


results_dir = Path('/mnt/c/Users/jwy/Desktop/21.12.30/')

dirs = ['0', '0.0001', '0.0005', '0.001', '0.002', '0.003', '0.004', '0.005']

for dir in dirs:
    result_dir_path = results_dir / dir / str('wp-resnet18_ae-cifar10-reconstruct-lambda_reconstruct_' + dir)
    result_path =  result_dir_path / 'console.log'

    print('---> analyse dir: {}'.format(dir))
    with open(result_path) as f:
        contents = f.readlines()

    # parse contents
    cla_loss_lists = []
    cla_acc_lists = []
    rec_loss_lists = []
    loss_lists = []

    auroc_lists = []
    aupr_in_lists = []
    aupr_out_lists = []
    fpr_tpr_lists = []

    pattern = r'\d+\.+\d*'
    aurocs = []
    aupr_ins = []
    aupr_outs = []
    fpr_tprs = []

    for line in contents:

        # epoch_count = 0
        if line.startswith('[cla_loss:'):
            # parse the digits
            cla_loss, cla_acc, rec_loss, train_loss = re.findall(pattern, line)
            cla_loss_lists.append(float(cla_loss))
            cla_acc_lists.append(float(cla_acc))
            rec_loss_lists.append(float(rec_loss))
            loss_lists.append(float(train_loss))
        elif line.startswith('[auroc:'):
            auroc, aupr_in, aupr_out, fpr_tpr = re.findall(pattern, line)
            aurocs.append(float(auroc))
            aupr_ins.append(float(aupr_in))
            aupr_outs.append(float(aupr_out))
            fpr_tprs.append(float(fpr_tpr))
        elif line.startswith('Epoch'):
            auroc_lists.append(sum(aurocs) / len(aurocs))
            aupr_in_lists.append(sum(aupr_ins) / len(aupr_ins))
            aupr_out_lists.append(sum(aupr_outs) / len(aupr_outs))
            fpr_tpr_lists.append(sum(fpr_tprs) / len(fpr_tprs))
            
            # clear
            aurocs = []
            aupr_ins = []
            aupr_outs = []
            fpr_tprs = []
    # parse over, visualize
    train_cla_loss_lists = cla_loss_lists[::2]
    test_cla_loss_lists = cla_loss_lists[1::2]
    visualize_loss(train_cla_loss_lists, test_cla_loss_lists, 'classification_loss', str(result_dir_path / 'classification_loss.png'))
    train_cla_acc_lists = cla_acc_lists[::2]
    test_cla_acc_lists = cla_acc_lists[1::2]
    visualize_loss(train_cla_acc_lists, test_cla_acc_lists, 'classification_acc', str(result_dir_path / 'classification_acc.png'))
    train_rec_loss_lists = rec_loss_lists[::2]
    test_rec_loss_lists = rec_loss_lists[1::2]
    visualize_loss(train_rec_loss_lists, test_rec_loss_lists, 'reconstruction_loss', str(result_dir_path / 'reconstruction_loss.png'))
    train_loss_lists = loss_lists[::2]
    test_loss_lists = loss_lists[1::2]
    visualize_loss(train_loss_lists, test_loss_lists, 'loss', str(result_dir_path / 'loss.png'))


    visualize_ood(auroc_lists, 'auroc', str(result_dir_path / 'auroc.png'))
    visualize_ood(aupr_in_lists, 'aupr_in', str(result_dir_path / 'aupr_in.png'))
    visualize_ood(aupr_out_lists, 'aupr_out', str(result_dir_path / 'aupr_out.png'))
    visualize_ood(fpr_tpr_lists, 'fpr_tpr', str(result_dir_path / 'fpr_tpr.png'))
