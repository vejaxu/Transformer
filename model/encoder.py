import torch
from torch import nn

from blocks.encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


if __name__ == '__main__':
    x = torch.randint(1, 10, size=(2, 4, 6))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(x.shape)
    encoder = Encoder(6, 100, 6, 12, 2, 4, 0.1, device)
    out = encoder(x, None)
    print(out.shape)