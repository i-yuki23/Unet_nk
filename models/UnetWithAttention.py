import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: ゲーティング信号のチャネル数（デコーダからのアップサンプル特徴）
            F_l: エンコーダのスキップ接続のチャネル数
            F_int: 中間チャネル数（一般的にはF_lやF_gより小さい）
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: ゲーティング信号 (デコーダからのアップサンプル特徴)
            x: エンコーダのスキップ接続特徴
        Returns:
            x に対する注意重み付け後の出力
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.3):
        super(UNetWithAttention, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, dropout_p)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Attention block for encoder 4
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = self.conv_block(1024, 512)  # 結合後のチャネルは512+512=1024

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = self.conv_block(512, 256)  # 256+256

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = self.conv_block(256, 128)  # 128+128

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = self.conv_block(128, 64)  # 64+64

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # (B, 64, H, W)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)  # (B, 128, H/2, W/2)
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)  # (B, 256, H/4, W/4)
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)  # (B, 512, H/8, W/8)
        pool4 = self.pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # (B, 1024, H/16, W/16)

        # Decoder with Attention
        up4 = self.upconv4(bottleneck)  # (B, 512, H/8, W/8)
        # Attention: encoder feature enc4
        att4 = self.att4(g=up4, x=enc4)
        dec4 = self.dec4(torch.cat((up4, att4), dim=1))  # (B, 1024, H/8, W/8)

        up3 = self.upconv3(dec4)  # (B, 256, H/4, W/4)
        att3 = self.att3(g=up3, x=enc3)
        dec3 = self.dec3(torch.cat((up3, att3), dim=1))  # (B, 512, H/4, W/4)

        up2 = self.upconv2(dec3)  # (B, 128, H/2, W/2)
        att2 = self.att2(g=up2, x=enc2)
        dec2 = self.dec2(torch.cat((up2, att2), dim=1))  # (B, 256, H/2, W/2)

        up1 = self.upconv1(dec2)  # (B, 64, H, W)
        att1 = self.att1(g=up1, x=enc1)
        dec1 = self.dec1(torch.cat((up1, att1), dim=1))  # (B, 128, H, W)

        return self.final(dec1)
