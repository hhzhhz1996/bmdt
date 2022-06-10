import torch
import torch.nn as nn
import torch.nn.functional as f


class GateCNN(nn.Module):

    def __init__(self, embedding_size=8, core_nums=128, drop_out=0.5, **params):
        super(GateCNN, self).__init__()
        self.params = params
        self.emb = nn.Embedding(257, embedding_size, padding_idx=0)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.conv1_1 = nn.Conv1d(embedding_size, core_nums, kernel_size=(512,), stride=(512,))
        self.conv1_2 = nn.Conv1d(embedding_size, core_nums, kernel_size=(512,), stride=(512,))
        self.conv2_1 = nn.Conv1d(embedding_size, core_nums, kernel_size=(1024,), stride=(512,))
        self.conv2_2 = nn.Conv1d(embedding_size, core_nums, kernel_size=(1024,), stride=(512,))

        self.fc = nn.Sequential(
            nn.BatchNorm1d(core_nums * 2),
            nn.Linear(core_nums*2, core_nums),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(core_nums, 1),
            nn.Sigmoid())

    def forward(self, x):
        # x shape   batch_size * 2M
        x_embed = self.emb(x)  # batch_size * 2M * 8
        x_embed = x_embed.permute(0, 2, 1)  # batch_size * 8 * 2M   
        x_embed = self.bn1(x_embed)
        c1 = self.sigmoid(self.conv1_1(x_embed)) * self.conv1_2(x_embed)  # batch_size * 128 * 4096
        c2 = self.sigmoid(self.conv2_1(x_embed)) * self.conv2_2(x_embed)  # batch_size * 128 * 4095
        out1 = torch.max(c1, -1)[0]
        out2 = torch.max(c2, -1)[0]
        out = torch.cat([out1, out2], -1)  # batch_size * 256
        return self.fc(out)


class LstmAttention(nn.Module):
    def __init__(self, **params):
        super(LstmAttention, self).__init__()
        self.params = params
        self.voc_size = params['voc_size']
        self.emb_size = params['emb_size']
        self.padding_index = params['padding_index']
        self.hidden_size = params['hidden_size']

        self.word_embeddings = nn.Embedding(self.voc_size, self.emb_size, self.padding_index)
        self.encoder = nn.LSTM(input_size=self.emb_size,
                               hidden_size=self.hidden_size,
                               batch_first=False,
                               num_layers=1)

        self.w_omega = nn.Parameter(torch.Tensor(
            self.hidden_size, self.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 1, bias=True),
            nn.Sigmoid()
        )

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        embeddings = self.word_embeddings(inputs)
        embeddings = embeddings.permute(1, 0, 2)  # step * batch_size * emb_size
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # outputs形状是(seq_len, batch_size, hidden_size)

        x = outputs.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, hidden_size)

        u = torch.tanh(torch.matmul(x, self.w_omega))  # u形状是(batch_size, seq_len, hidden_size)
        att = torch.matmul(u, self.u_omega)  # att形状是(batch_size, seq_len, 1)
        att_score = f.softmax(att, dim=1)  # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score  # scored_x形状是(batch_size, seq_len, hidden_size)
        feat = torch.sum(scored_x, dim=1)  # feat形状是(batch_size, hidden_size)
        outs = self.decoder(feat)  # out形状是(batch_size, 1)
        return outs
