import torch
from torch import nn
from torch.nn import functional as F
import math


# For CNN
class SimpleCNN(nn.Module):
    default_hparams = {
        "num_conv_layers": 3,
        "num_fc_layers": 2,
        "conv_channels": 128,
        "fc_units": 512
    }
    def __init__(self, hparams, num_classes=10, device="cpu"):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        num_conv_layers = hparams["num_conv_layers"]
        num_fc_layers = hparams["num_fc_layers"]
        conv_channels = hparams["conv_channels"]
        fc_units = hparams["fc_units"]
        
        # Convolutional layers
        in_channels = 3
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = conv_channels
        
        # Fully connected layers
        fc_input = conv_channels * (32 // (2**num_conv_layers))**2
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input, fc_units))
            self.fc_layers.append(nn.ReLU())
            fc_input = fc_units
        self.fc_layers.append(nn.Linear(fc_input, num_classes))
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x


# LSTM
class Encoder(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """
        - x: (B, W, 1)
        - h_n: (L, B, H)
        - c_n: (L, B, H)
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, input_size=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_c):
        """
        - x: (B, 1, input_size)
        - h_c: (L, B, H)
        - output: (B, 1, H)
        - pred: (B, 1, output_size)
        """
        output, (h, c) = self.lstm(x, h_c)
        pred = self.fc(output)
        return pred, (h, c)


class LSTM_Seq2Seq(nn.Module):
    default_hparams = {
        "hidden_size": 128,
        "num_layers": 3,
    }
    def __init__(self, hparams, pred=10, input_size=1, output_size=1, device='cpu'):
        super().__init__()

        hidden_size = hparams["hidden_size"]
        num_layers  = hparams["num_layers"]

        self.encoder = Encoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_size=input_size,
        )
        self.decoder = Decoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_size=input_size,
            output_size=output_size,
        )
        self.pred_len = pred
        self.device = device

    def forward(self, x):
        B, _, F = x.shape

        # Encoding
        h_c = self.encoder(x)

        # Predict
        pred_output = []
        pred_input  = torch.zeros((B, 1, F)).to(self.device)
        for _ in range(self.pred_len):
            pred_input, h_c = self.decoder(pred_input, h_c)
            pred_output.append(pred_input)
        pred_output = torch.cat(pred_output, dim=1)

        return pred_output


def create_net(sizes):
    net = []
    for i in range(len(sizes)-1):
        net.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            net.append(nn.GELU())
    return nn.Sequential(*net)


class DeepONet(nn.Module):
    default_hparams = {
        "hidden_size": 1024,
        "num_layers": 6,
        "num_branch": 50,
    }
    def __init__(self, hparams, device="cpu"):
        super().__init__()

        self.branch_net = create_net(
            [100]
            + [hparams["hidden_size"]]*(hparams["num_layers"]-1)
            + [hparams["num_branch"]]
        )
        self.trunk_net = create_net(
            [1]
            + [hparams["hidden_size"]]*(hparams["num_layers"]-1)
            + [hparams["num_branch"]]
        )
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True).to(device)
        self.device = device

    def forward(self, u, y):
        window = y.shape[1]
        branch_out = self.branch_net(u)
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1])
                                for i in range(window)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        - x: (B, W, d_model)
        - self.pe: (1, M, d_model)
        - self.pe[:, :x.size(1), :]: (1, W, d_model)
        - output: (B, W, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

    def forward(self, x):
        """
        - x: (B, W1, 1)
        - x (after embedding): (B, W1, d_model)
        - out: (B, W1, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TFDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, memory):
        """
        - x: (B, W2, 1)
        - x (after embedding): (B, W2, d_model)
        - memory: (B, W1, d_model)
        - out: (B, W2, d_model)
        - out (after fc): (B, W2, 1)
        - out (after squeeze): (B, W2)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_decoder(x, memory)
        out = self.fc(out)
        out = out.squeeze(-1)
        return out


class TFONet(nn.Module):
    default_hparams = {
        "d_model": 128,
        "nhead": 2,
        "dim_feedforward": 512,
        "num_layers": 3,
        "dropout": 0.0
    }
    def __init__(self, hparams, device="cpu"):
        super().__init__()

        d_model = hparams["d_model"]
        nhead = hparams["nhead"]
        num_layers = hparams["num_layers"]
        dim_feedforward = hparams["dim_feedforward"]
        dropout = hparams["dropout"]

        self.branch_net = TFEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout)
        self.trunk_net = TFDecoder(
            d_model, nhead, num_layers, dim_feedforward, dropout)
        self.device = device

    def forward(self, u, y):
        """
        - u: (B, W1)
        - y: (B, W2)
        - u (after reshape): (B, W1, 1)
        - y (after reshape): (B, W2, 1)
        - memory: (B, W1, d_model)
        - o: (B, W2)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        memory = self.branch_net(u)

        # Decoding
        o = self.trunk_net(y, memory)
        return o
