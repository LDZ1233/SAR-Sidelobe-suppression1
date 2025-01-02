import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """U-Net style double convolution block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """Enhanced encoder with U-Net style double convolutions"""

    def __init__(self, input_channels=1):
        super().__init__()
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, [x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    """U-Net style decoder block with skip connections"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handling cases where the dimensions don't match exactly
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """Enhanced decoder with U-Net style upsampling"""

    def __init__(self, num_classes=1):
        super().__init__()
        self.up1 = DecoderBlock(1024, 512, 512)
        self.up2 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up4 = DecoderBlock(128, 64, 64)
        self.outc = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[3])
        x = self.up2(x, skip_features[2])
        x = self.up3(x, skip_features[1])
        x = self.up4(x, skip_features[0])
        return self.outc(x)


class ChangeDetectionNetwork(nn.Module):
    """Enhanced Change Detection Network with U-Net architecture"""

    def __init__(self, input_channels=1, num_classes=1):
        super().__init__()
        self.encoder1 = Encoder(input_channels)
        self.encoder2 = Encoder(input_channels)

        # Feature fusion layers for skip connections
        self.skip_fusions = nn.ModuleList([
            DoubleConv(128, 64),  # For skip1
            DoubleConv(256, 128),  # For skip2
            DoubleConv(512, 256),  # For skip3
            DoubleConv(1024, 512)  # For skip4
        ])

        # Bottleneck fusion
        self.fusion = DoubleConv(2048, 1024)

        self.decoder = Decoder(num_classes)

    def _fuse_features(self, feat1, feat2):
        return self.fusion(torch.cat([feat1, feat2], dim=1))

    def _merge_skip_features(self, skip1, skip2):
        merged = []
        for i, (f1, f2) in enumerate(zip(skip1, skip2)):
            fused = self.skip_fusions[i](torch.cat([f1, f2], dim=1))
            merged.append(fused)
        return merged

    def forward(self, x1, x2):
        # Encode both images
        feat1, skip1 = self.encoder1(x1)
        feat2, skip2 = self.encoder2(x2)

        # Fuse bottleneck features
        fused_features = self._fuse_features(feat1, feat2)

        # Merge skip connections
        merged_skips = self._merge_skip_features(skip1, skip2)

        # Decode to produce change mask
        change_mask = self.decoder(fused_features, merged_skips)
        return change_mask


def test_network():
    model = ChangeDetectionNetwork(input_channels=1)
    batch_size = 1
    x1 = torch.randn(batch_size, 1, 256, 256)
    x2 = torch.randn(batch_size, 1, 256, 256)
    output = model(x1, x2)
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {output.shape}")
    return output


if __name__ == "__main__":
    test_network()