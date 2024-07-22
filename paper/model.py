import torch
from torch import nn
from torch.nn import functional as F
from vit_pytorch import SimpleViT


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


class ViT(nn.Module):
    default_hparams = {
        "dim": 32,
        "heads": 2,
        "mlp_dim": 512,
        "depth": 3,
    }
    def __init__(self, hparams, num_classes=10, device="cpu"):
        super(ViT, self).__init__()
        self.model = SimpleViT(
            image_size = 32,
            patch_size = 4,
            num_classes = num_classes,
            dim = hparams["dim"],
            heads = hparams["heads"],
            mlp_dim = hparams["mlp_dim"],
            depth = hparams["depth"],
        )

    def forward(self, x):
        return self.model(x)


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
