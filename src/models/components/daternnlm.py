from typing import Tuple

import torch
from torch import Tensor, nn

from src.models.components.masks import make_masks


class DateRNNLM(nn.Module):
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
        context_dim=256,
        embedding_dim=128,
        num_date_embeddings=1234,
        dropout=0.15,
        rnn_layers=2,
        rnn_neurons=1024,
        dnn_neurons=256,
        nhead=6,
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_neurons, embedding_dim, padding_idx=0)
        self.date_embedding = nn.Embedding(num_date_embeddings, context_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=embedding_dim + context_dim,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.d_atten = nn.MultiheadAttention(
            embed_dim=context_dim,
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

    def forward(self, x: Tuple[Tensor, Tensor], hx=None) -> Tensor:
        src, date_contexts = x
        src_mask, src_key_padding_mask = make_masks(src)  # (B,T), (B,T)
        src = self.embedding(src)  # (B,T,E)
        src = self.dropout(src)
        date_contexts = self.date_embedding(date_contexts)
        if len(date_contexts.shape) != 2:
            date_contexts = date_contexts.mean(-2)  # or mean (B, context_dim)

        # If 2d tensor, add a time-axis
        # This is used for inference time
        if len(src.shape) == 2:
            src = src.unsqueeze(dim=1)
            self.reshape = True

        date_contexts = date_contexts.unsqueeze(1)  # (B,1,context_dim)
        date_contexts = date_contexts.repeat(1, src.shape[1], 1)  # (B,T,context_dim)

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
