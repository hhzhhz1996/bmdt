import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
from features import get_byte_stream_single

import linecache


class DataSet(Dataset):
    # for api model

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
        return features, labels

    def __call__(self, batch):
        return self.pad(batch)


class DataSet2(Dataset):
    # for byte stream model

    def __init__(self, path, **params):
        self.path = path
        malicious_dir_path = linecache.getlines(os.path.join(path, 'malware'))
        benign_dir_path = linecache.getlines(os.path.join(path, 'benign'))

        malware = []
        for file in os.listdir(malicious_dir_path):
            malware.append(get_byte_stream_single(os.path.join(malicious_dir_path, file)))

        benign = []
        for file in os.listdir(benign_dir_path):
            benign.append(get_byte_stream_single(os.path.join(benign_dir_path, file)))

        self.feature = malware + benign
        len1, len2 = len(malware), len(benign)
        print(len1)
        print(len2)
        self.len = len1 + len2
        self.labels = np.array([1 if i < len1 else 0 for i in range(len1+len2)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        features = torch.tensor(self.feature[index], dtype=torch.long)
        labels = torch.tensor(self.labels, dtype=torch.float)  # BCE-loss is float   CrossEntropyLoss is long
        return features, labels
