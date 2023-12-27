from torch import Tensor, nn


class RNNLM(nn.Module):
    def __init__(
        self,
        output_neurons: int,
        embedding_dim=128,
        dropout=0.2,
        rnn_neurons=1024,
        rnn_layers=2,
        # dnn_neurons=256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(output_neurons, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            batch_first=True,
        )
        # self.dnn = nn.Sequential(
        #     nn.Linear(rnn_neurons, dnn_neurons),
        #     nn.LayerNorm(dnn_neurons),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=dropout),
        # )
        self.out = nn.Linear(rnn_neurons, output_neurons)
        self.reshape = False

    def forward(self, x: Tensor, hx=None) -> Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            self.reshape = True

        x, _ = self.rnn(x, hx)
        # x = self.dnn(x)
        out = self.out(x)

        if self.reshape:
            out = out.squeeze(dim=1)

        return out


if __name__ == "__main__":
    _ = RNNLM()
