import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.masks import make_masks


class NormalizedEmbedding(nn.Module):
    """This class implements the normalized embedding layer for the transformer.

    Since the dot product of the self-attention is always normalized by sqrt(d_model)
    and the final linear projection for prediction shares weight with the embedding layer,
    we multiply the output of the embedding by sqrt(d_model).

    Arguments
    ---------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    vocab: int
        The vocab size.

    Example
    -------
    >>> emb = NormalizedEmbedding(512, 1000)
    >>> trg = torch.randint(0, 999, (8, 50))
    >>> emb_fea = emb(trg)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor):
        """Processes the input tensor x and returns an output tensor."""
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size: int, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        ---------
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class TransformerLM(nn.Module):
    """This is an implementation of transformer language model.

    The architecture is based on the paper "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead : int
        The number of heads in the multiheadattention models (default=8).
    num_encoder_layers : int
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int
        The dimension of the feedforward network model (default=2048).
    dropout : int
        The dropout value (default=0.1).
    activation: torch class
        The activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
    decoder_use_memory: bool
        whether to use the hidden state in the decoder

    Example
    -------
    >>> src = torch.randint(0, 720, [8, 120])
    >>> net = TransformerLM(720, 512, 8, 1, 0, 1024, activation=torch.nn.GELU)
    >>> enc_out = net.forward(src)
    >>> print(enc_out.shape)
    torch.Size([8, 120, 720])
    """

    def __init__(
        self,
        vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        d_embedding=None,
        max_length=2500,
        num_date_embeddings=1234,
        date_context_range=7,
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_length)

        self.d_embedding = d_embedding
        if d_embedding is None:
            self.d_embedding = d_model

        self.custom_src_module = NormalizedEmbedding(self.d_embedding, vocab)
        self.date_embedding = nn.Embedding(num_date_embeddings, self.d_embedding, padding_idx=0)
        self.date_context_range = date_context_range

        self.embedding_proj = None
        if d_embedding is not None:
            self.embedding_proj = nn.Linear(self.d_embedding, d_model)

        self.output_proj = nn.Linear(d_model, vocab)
        # self.demb_proj = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, d_model),
        # )
        self.demb_proj = nn.Linear(d_model, d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_ffn,
            dropout,
            activation=activation,
            norm_first=normalize_before,
            layer_norm_eps=1e-6,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # reset the params of the transformer model
        self._reset_params()

    def pre_compute_future_date_embedding(self, max_date):
        all_date_embeddings = torch.zeros(
            (max_date + 1, self.date_embedding.embedding_dim),
            device=self.date_embedding.weight.device,
        )
        all_date_embeddings[: self.date_embedding.num_embeddings] = self.date_embedding.weight

        # auto regressive generate date embedding
        for i in range(self.date_embedding.num_embeddings, max_date):
            tmp_date_context = torch.arange(
                i - (self.date_context_range - 1), i, device=all_date_embeddings.device
            ).unsqueeze(0)

            src_mask, src_key_padding_mask = make_masks(tmp_date_context)
            src_key_padding_mask[0, :] = False

            tmp_date_context = all_date_embeddings[tmp_date_context]

            src = tmp_date_context + self.positional_encoding(tmp_date_context)

            encoder_out = self.transformer_encoder(
                src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True
            )
            pred_demb = self.demb_proj(encoder_out[:, -1, :])
            all_date_embeddings[i] = pred_demb

        return nn.Embedding(
            max_date + 1,
            self.date_embedding.embedding_dim,
            padding_idx=self.date_embedding.padding_idx,
            _weight=all_date_embeddings,
        ).to(self.date_embedding.weight.device)

    def inference_forward(self, x):
        src, date_contexts = x

        date_contexts = date_contexts.unsqueeze(1)
        max_date = self.date_embedding.num_embeddings - 1
        date_embedding = self.date_embedding
        if date_contexts.max() > max_date:
            print(" > max_date")
            print("predict future date embedding")
            date_embedding = self.pre_compute_future_date_embedding(date_contexts.max())

        date_contexts = date_contexts.repeat(1, self.date_context_range)
        date_contexts = date_contexts - torch.arange(
            self.date_context_range, device=date_contexts.device
        ).flip(0)  # old to new
        date_contexts = date_contexts.clamp(min=0)

        src_mask, src_key_padding_mask = make_masks(torch.cat([date_contexts, src], dim=1))
        src_key_padding_mask[F.pad(date_contexts, (0, src.size(1)), value=1) == 0] = False

        src = self.custom_src_module(src)
        if self.embedding_proj is not None:
            src = self.embedding_proj(src)
        date_contexts = date_embedding(date_contexts)

        src = torch.cat([date_contexts, src], dim=1)
        src = src + self.positional_encoding(src)
        # src = torch.cat([date_contexts, src], dim=1)

        encoder_out = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True
        )

        encoder_out = encoder_out[:, self.date_context_range :, :]

        pred = self.output_proj(encoder_out)
        pred_demb = self.demb_proj(encoder_out[:, self.date_context_range - 2, :])

        return (
            pred,
            date_contexts[:, -1, :].detach(),
            pred_demb,
        )

    def forward(self, x, hx=None):
        """
        ---------
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        """
        src, date_contexts = x

        date_contexts = date_contexts.unsqueeze(1)
        max_date = self.date_embedding.num_embeddings - 1
        if date_contexts.max() > max_date:
            print(" > max_date auto clamped")
        date_contexts = date_contexts.clamp(max=max_date)

        date_contexts = date_contexts.repeat(1, self.date_context_range)
        date_contexts = date_contexts - torch.arange(
            self.date_context_range, device=date_contexts.device
        ).flip(0)  # old to new
        date_contexts = date_contexts.clamp(min=0)

        src_mask, src_key_padding_mask = make_masks(torch.cat([date_contexts, src], dim=1))
        src_key_padding_mask[F.pad(date_contexts, (0, src.size(1)), value=1) == 0] = False

        src = self.custom_src_module(src)
        if self.embedding_proj is not None:
            src = self.embedding_proj(src)
        date_contexts = self.date_embedding(date_contexts)

        # stop gradient on old date emb
        date_contexts[:, : self.date_context_range - 1, :] = date_contexts[
            :, : self.date_context_range - 1, :
        ].detach()

        src = torch.cat([date_contexts, src], dim=1)
        src = src + self.positional_encoding(src)
        # src = torch.cat([date_contexts, src], dim=1)

        encoder_out = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True
        )

        # mlp version
        pred_demb = self.demb_proj(encoder_out[:, self.date_context_range - 2, :])
        # no mlp version
        # pred_demb = encoder_out[:, self.date_context_range - 2, :]

        encoder_out = encoder_out[:, self.date_context_range :, :]

        pred = self.output_proj(encoder_out)

        return (
            pred,
            date_contexts[:, -1, :].detach(),
            pred_demb,
        )

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
