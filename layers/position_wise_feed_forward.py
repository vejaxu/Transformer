import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 4, 6)
    print(x.shape)
    ffn = PositionwiseFeedForward(x.shape[-1], x.shape[-1] * 2)
    out = ffn(x)
    print(out.shape)