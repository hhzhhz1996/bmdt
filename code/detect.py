from features import *
import numpy as np
import torch

from constant import cnn as byte_stream_model
from constant import rnn as api_model
from constant import shell_detect_model


def shell_detect(file_path, pe):
    feature = get_feature_shell(file_path, pe)

    array = np.array(feature).reshape(1, -1)
    result = shell_detect_model.predict_proba(array)

    return result, feature


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
