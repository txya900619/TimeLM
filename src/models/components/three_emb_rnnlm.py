from typing import Tuple

import torch
from torch import Tensor, nn

from src.models.components.masks import make_masks


class SelfAttentivePooling(nn.Module):
    def __init__(self, h_size=3, w_size=128, patch_size=3, nhead=3, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        # self.cnn = nn.Conv2d(in_channels=1, out_channels=patch_size, kernel_size=patch_size,stride=1, padding='same')
        # self.cnn_activation = nn.ReLU()
        self.patch_embeddings = nn.Conv2d(
            in_channels=1, out_channels=patch_size, kernel_size=patch_size, stride=patch_size
        )
        self.post_patch_norm = nn.BatchNorm2d(patch_size)
        self.activation = nn.ReLU()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, h_size * w_size // (patch_size**2), patch_size)
        )
        self.atten = nn.MultiheadAttention(
            num_heads=nhead,
            embed_dim=patch_size,
            dropout=dropout,
        )

        # self.linear = nn.Linear(h_size*w_size // (patch_size**2), h_size * w_size)

        self.conv1x1 = nn.Conv2d(patch_size, 1, 1, 1)
        self.post_conv1x1_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=(h_size, 2))

    def forward(self, x: Tensor) -> Tensor:
        origin_x = x

        x = x.unsqueeze(1)  # (n, h, w) -> (n, 1, h, w)

        # x = self.cnn(x)
        # x = self.cnn_activation(x)

        x = self.patch_embeddings(x)  # (n, 1, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.activation(self.post_patch_norm(x))
        x = x.reshape(
            x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
        )  # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.permute(0, 2, 1)  # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)

        x = x + self.pos_embed

        x = x.permute(0, 1, 2)

        x, _ = self.atten(x, x, x)

        x = x.permute(0, 2, 1)  # (n, (n_h * n_w), hidden_dim) -> (n, hidden_dim, (n_h * n_w))
        # x = self.linear(x) # (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, (h * w))
        x = x.reshape(
            x.shape[0], x.shape[1], 1, x.shape[2]
        )  # (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, n_h, n_w)
        x = nn.functional.interpolate(
            x, size=(3, 128), mode="bilinear"
        )  # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, h, w)

        x = self.conv1x1(x)  # (n, hidden_dim, h, w) -> (n, 1, h, w)
        x = self.sigmoid(self.post_conv1x1_norm(x))
        x = x.squeeze(1)  # (n, 1, h, w) -> (n, h, w)

        x = origin_x * x

        return self.pool(x)


class ThreeEmbRNNLM(nn.Module):
    """This model is a combination of embedding layer, RNN, DNN. It can be used for RNNLM.

    Arguments
    ---------
    output_neurons : int
        Number of entries in embedding table, also the number of neurons in
        output layer.
    embedding_dim : int
        Size of embedding vectors (default 128).
    activation : torch class
        A class used for constructing the activation layers for DNN.
    dropout : float
        Neuron dropout rate applied to embedding, RNN, and DNN.
    rnn_class : torch class
        The type of RNN to use in RNNLM network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_re_init : bool
        Whether to initialize rnn with orthogonal initialization.
    rnn_return_hidden : bool
        Whether to return hidden states (default True).
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> model = RNNLM(output_neurons=5)
    >>> inputs = torch.Tensor([[1, 2, 3]])
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([1, 3, 5])
    """

    def __init__(
        self,
        output_neurons: int,
        embedding_dim=128,
        num_date_embeddings=1234,
        dropout=0.15,
        rnn_layers=2,
        rnn_neurons=1024,
        dnn_neurons=256,
        nhead=6,
        year_dim=128,
        month_dim=128,
        day_dim=128,
        num_year_embeddings=51,
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_neurons, embedding_dim, padding_idx=0)

        # idx 0 is reserved for unknown token
        self.year_embedding = nn.Embedding(num_year_embeddings, year_dim)
        self.month_embedding = nn.Embedding(13, month_dim)
        self.day_embedding = nn.Embedding(32, day_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.pooling = SelfAttentivePooling()
        self.rnn = nn.LSTM(
            input_size=embedding_dim + (year_dim + month_dim + day_dim) // 6,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.d_atten = nn.MultiheadAttention(
            embed_dim=(year_dim + month_dim + day_dim) // 6,
            num_heads=nhead,
            dropout=dropout,
            kdim=embedding_dim,
            vdim=embedding_dim,
            batch_first=True,
        )
        self.dnn = nn.Sequential(
            nn.Linear(rnn_neurons, dnn_neurons),
            nn.LayerNorm(dnn_neurons),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )
        self.out = nn.Linear(dnn_neurons, output_neurons)
        self.reshape = False

    def forward(self, x: Tuple[Tensor, Tensor, Tensor, Tensor], hx=None) -> Tensor:
        src, year, month, day = x

        src_mask, src_key_padding_mask = make_masks(src)  # (B,T), (B,T)
        src = self.embedding(src)  # (B,T,E)
        src = self.dropout(src)

        year_contexts = self.year_embedding(year).unsqueeze(1)  # (B,1,year_dim)
        month_contexts = self.month_embedding(month).unsqueeze(1)
        day_contexts = self.day_embedding(day).unsqueeze(1)

        # If 2d tensor, add a time-axis
        # This is used for inference time
        if len(src.shape) == 2:
            src = src.unsqueeze(dim=1)
            self.reshape = True

        date_contexts = torch.concat(
            (year_contexts, month_contexts, day_contexts), dim=1
        )  # (B, 3, year_dim)

        date_contexts = self.pooling(date_contexts)
        date_contexts = date_contexts.repeat(1, src.shape[1], 1)

        date_contexts, _ = self.d_atten(
            date_contexts, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )

        src = torch.cat([src, date_contexts], dim=-1)  # (B,T,E+context_dim)

        src, _ = self.rnn(src, hx)
        src = self.dnn(src)
        out: Tensor = self.out(src)

        if self.reshape:
            out = out.squeeze(dim=1)

        return out
