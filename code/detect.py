import os

from features import *
import numpy as np
import torch

from constant import cnn as byte_stream_model
from constant import rnn as api_model
from constant import shell_detect_model


def shell_detect(file_path):
    feature = get_pack_check_features(file_path)
    array = np.array(feature).reshape(1, -1)
    result = shell_detect_model.predict_proba(array)
    return result[0][0], result[0][1]


def api_detect(file_path):
    feature = get_api_seq_single(file_path, to_id=True)
    array = torch.tensor(feature)
    array = array.unsqueeze(0)
    result = api_model(array)
    return result


def byte_stream_detect(file_path):
    feature = get_byte_stream_single(file_path)
    array = torch.tensor(feature)
    array = array.unsqueeze(0)
    result = byte_stream_model.forward(array)
    return result


def detect(file_path):
    p_unpack, p_packed = shell_detect(file_path)
    p_api = api_detect(file_path).item()
    p_bytes = byte_stream_detect(file_path).item()
    return p_unpack * p_api + p_packed * p_bytes


if __name__ == '__main__':
    d = 'C:/users/shini/desktop/malware'
    for file in os.listdir(d):
        path = os.path.join(d, file)
        print(byte_stream_detect(path))
