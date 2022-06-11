import joblib
import torch
from model import GateCNN, LstmAttention

BYTE_STREAM_LENGTH = 2 * 1024 * 1024

api_mapping = {}
with open('../resources/api.txt') as f:
    for idx, api_name in enumerate(f.readlines()):
        api_name = api_name.strip()
        api_mapping[api_name] = idx + 1  # index 0 reserves for padding
        api_mapping[str(idx + 1)] = api_name


shell_detect_model = joblib.load('../resources/RF.model')

cnn = GateCNN().to(device)
state = torch.load('../resources/CNN.model', map_location=torch.device(device))
cnn.load_state_dict(state['state_dict'])
byte_stream_model = cnn.eval()

rnn = LstmAttention(voc_size=voc_size, emb_size=emb_size, padding_index=padding_index, hidden_size=hidden_size)\
    .to(device)
state = torch.load('../resources/LSTM.model', map_location=torch.device(device))
rnn.load_state_dict(state['state_dict'])
api_model = rnn.eval()
