import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np


import linecache


class DataSet(Dataset):

    def __init__(self, path, **params):
        self.path = path
        malicious = linecache.getlines(os.path.join(path, 'malware.txt'))
        benign = linecache.getlines(os.path.join(path, 'benign.txt'))

        self.feature = malicious + benign
        len1, len2 = len(malicious), len(benign)
        print(len1)
        print(len2)
        self.len = len1 + len2
        self.labels = np.array([1 if i < len1 else 0 for i in range(len1+len2)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        feature_list = list(map(int, self.feature[index].split()))
        return feature_list, self.labels[index]


class PadCollate:
    
    def __init__(self):
        self.padding_index = 0
        self.cate_padding = 0

    def pad(self, batch):
        longest = 0
        
        for line, label in batch:
            longest = max(longest, len(line))

        features = []
        labels = []

        for line, label in batch:
            if len(line) < longest:
                for i in range(longest-len(line)):
                    line.append(self.padding_index)

            features.append(line)
            labels.append(label)

        features = torch.tensor(features, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)  # BCE-loss is float   CrossEntropyLoss is long
        return features, labels,

    def __call__(self, batch):
        return self.pad(batch)

