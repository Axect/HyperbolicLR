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
    def __init__(self, hparams, device="cpu"):
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
        self.fc_layers.append(nn.Linear(fc_input, 10))
    
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
    def __init__(self, hparams, device="cpu"):
        super(ViT, self).__init__()
        self.model = SimpleViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = hparams["dim"],
            heads = hparams["heads"],
            mlp_dim = hparams["mlp_dim"],
            depth = hparams["depth"],
        )

    def forward(self, x):
        return self.model(x)
