import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.3):
        super(UNetPlusPlus, self).__init__()

        self.pool = nn.MaxPool2d(2)

        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, dropout_p)

        # Dense Skip Connections in Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_1 = self.conv_block(1024, 512)  # 512 (upconv) + 512 (skip)
        self.dec4_2 = self.conv_block(1024, 512)  # 512 (dec4_1) + 512 (skip)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_1 = self.conv_block(512, 256)   # 256 (upconv) + 256 (skip)
        self.dec3_2 = self.conv_block(1024, 256)  # 256 (dec3_1) + 512 (dec4_2) + 256 (skip)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_1 = self.conv_block(256, 128)   # 128 (upconv) + 128 (skip)
        self.dec2_2 = self.conv_block(512, 128)   # 128 (dec2_1) + 256 (dec3_2) + 128 (skip)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_1 = self.conv_block(128, 64)    # 64 (upconv) + 64 (skip)
        self.dec1_2 = self.conv_block(256, 64)    # 64 (dec1_1) + 128 (dec2_2) + 64 (skip)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))  # Dropoutを追加
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder with dense skip connections
        dec4_1 = self.dec4_1(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec4_2 = self.dec4_2(torch.cat([dec4_1, enc4], dim=1))

        dec3_1 = self.dec3_1(torch.cat([self.upconv3(dec4_2), enc3], dim=1))
        dec3_2 = self.dec3_2(torch.cat([dec3_1, F.interpolate(dec4_2, size=dec3_1.shape[2:], mode='bilinear', align_corners=True), enc3], dim=1))

        dec2_1 = self.dec2_1(torch.cat([self.upconv2(dec3_2), enc2], dim=1))
        dec2_2 = self.dec2_2(torch.cat([dec2_1, F.interpolate(dec3_2, size=dec2_1.shape[2:], mode='bilinear', align_corners=True), enc2], dim=1))

        dec1_1 = self.dec1_1(torch.cat([self.upconv1(dec2_2), enc1], dim=1))
        dec1_2 = self.dec1_2(torch.cat([dec1_1, F.interpolate(dec2_2, size=dec1_1.shape[2:], mode='bilinear', align_corners=True), enc1], dim=1))

        # Final output
        return self.final(dec1_2)
