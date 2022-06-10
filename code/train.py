import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import argparse
from data import DataSet, PadCollate
from model import LstmAttention


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--emb_dim', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--device', default='cuda', type=str)


args = parser.parse_args()
print(args)

epochs = args.epochs
batch_size = args.batch_size
emb_dim = args.emb_dim
learning_rate = args.lr
device = args.device
feature_dim = 16974  # 16975 as unknown   num of api is 16974
padding_index = 0
hidden_size = 64
weight_decay = 0.0001

dataset_path = '/home/LAB/huangz19/rnn/dataset-tmp/'
save_path = '/home/LAB/huangz19/rnn/model/'


def cur_time():
    return str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def save_checkpoint(state, model_name):
    file_name = model_name + f'_model_best' + '.model'
    model_path = os.path.join(save_path, file_name)
    torch.save(state, model_path)


def main_work():
    print(device)
    model = LstmAttention(voc_size=feature_dim+1, emb_size=emb_dim,
                          padding_index=padding_index, hidden_size=hidden_size).to(device)

    print(model.params)
    print(model)
    total_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(total_params)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(optim)

    loss_fn = nn.BCELoss()
    print(loss_fn)

    dataset = DataSet(dataset_path)
    len_1 = int(len(dataset) * 0.8)
    len_2 = int(len(dataset) * 0.1)
    len_3 = len(dataset) - len_1 - len_2

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [len_1, len_2, len_3])
    
    print('train_set:', len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PadCollate(padding_index=padding_index),
    )

    print('valid_set:', len(valid_set))
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PadCollate(padding_index=padding_index),
    )
    
    print('test_set:', len(test_set))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PadCollate(padding_index=padding_index),
    )

    best_valid_loss = 100
    best_test_loss = 100
    best_epoch = 0
    for epoch in range(epochs):

        model.train()
        print('%s epoch[%d]\t training...' % (cur_time(), epoch))
        mean_loss = 0
        for index, (features, labels) in enumerate(train_loader):

            optim.zero_grad()
            features, labels = Variable(features.to(device)), Variable(labels.to(device))
            out = model(features).squeeze()
            loss = loss_fn(out, labels)
            loss.backward()
            optim.step()
            
            mean_loss += loss.item()

        mean_loss /= index + 1
        print('%s epoch[%d]\t train-loss:\t%f' % (cur_time(), epoch, mean_loss))

        model.eval()
        mean_loss = 0
        for index, (features, labels) in enumerate(valid_loader):
            features, labels = features.to(device), labels.to(device)
            out = model(features).squeeze()
            loss = loss_fn(out, labels)
            mean_loss += loss.item()

        mean_loss /= index + 1
        print('%s epoch[%d]\t valid-loss:\t%f' % (cur_time(), epoch, mean_loss))

        if mean_loss < best_valid_loss:
            best_valid_loss = mean_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optim
            }, model.__class__.__name__)
        
            mean_loss = 0
            tp, tn, fp, fn = 0, 0, 0, 0

            for index, (features, labels) in enumerate(test_loader):
                features, labels = features.to(device), labels.to(device)
                out = model(features).squeeze()
                loss = loss_fn(out, labels)
                mean_loss += loss.item()

                predict = out > 0.5

                equals = (predict == labels)

                tmp_tp = (equals & (labels == 1)).sum().item()
                tmp_tn = equals.sum().item() - tmp_tp
                tmp_fn = (~equals & (labels == 1)).sum().item()
                tmp_fp = ((~equals).sum().item()) - tmp_fn

                tp += tmp_tp
                tn += tmp_tn
                fp += tmp_fp
                fn += tmp_fn

                mean_loss /= index + 1
                print('%s epoch[%d]\t test-loss:\t%f' % (cur_time(), epoch, mean_loss))

            print(f'tp:\t{tp} \ntn:\t{tn} \nfp:\t{fp} \nfn:\t{fn}')
            recall = tp / (tp + fn)
            precise = tp / (tp + fp)
            f1 = 2 * recall * precise / (recall + precise)
            print(f'recall:\t{recall}')
            print(f'precise:\t{precise}')
            print(f'f1:\t{f1}')

    print('%s best performance: epoch[%d]\t eval-loss: %f \t test-loss: %f' % (cur_time(), best_epoch, best_valid_loss, best_test_loss))


if __name__ == '__main__':
    main_work()
