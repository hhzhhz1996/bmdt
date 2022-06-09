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
    return result


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
    p_packed = shell_detect(file_path)
    p_unpack = 1 - p_packed
    return p_packed * byte_stream_detect(file_path) + p_unpack * api_detect(file_path)
