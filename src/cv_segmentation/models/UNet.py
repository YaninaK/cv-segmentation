import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Additional layers in encoder
        self.enc_conv1_additional = nn.Conv2d(32, 32, 3, padding=1)
        self.enc_conv2_additional = nn.Conv2d(64, 64, 3, padding=1)

        # Dropout layers
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.enc_conv1(x))
        x1 = torch.relu(self.enc_conv1_additional(x1))
        x2 = self.maxpool(x1)
        x2 = torch.relu(self.enc_conv2(x2))
        x2 = torch.relu(self.enc_conv2_additional(x2))
        x2 = self.dropout(x2)
        x3 = self.maxpool(x2)
        x3 = torch.relu(self.enc_conv3(x3))
        x3 = torch.relu(self.enc_conv4(x3))
        x3 = self.dropout(x3)

        # Decoder
        x = torch.relu(self.upconv1(x3))
        x = torch.cat([x2, x], dim=1)
        x = torch.relu(self.dec_conv1(x))
        x = torch.relu(self.upconv2(x))
        x = torch.cat([x1, x], dim=1)
        x = torch.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.final_conv(x))

        return x
