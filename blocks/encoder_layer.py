import torch
from torch import nn

# from layers.layer_norm import LayerNorm
# from layers.multi_head_attention import MultiHeadAttention
# from layers.position_wise_feed_forward import PositionwiseFeedForward
from layers import LayerNorm, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x, _ = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


if __name__ == '__main__':
    x = torch.randn(size=(2, 4, 6), dtype=torch.float32)
    print(x.shape)
    encoder = EncoderLayer(6, 12, 3, 0.1)
    out = encoder(x, None)
    print(out.shape)