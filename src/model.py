import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CustomUNet(nn.Module):
    """Pure PyTorch standard U-Net architecture fallback."""
    def __init__(self, in_channels=1, classes=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return self.sigmoid(logits)


def build_model(cfg):
    """
    Factory function to build segmentation model according to config.
    """
    m_cfg = cfg["model"]
    arch = m_cfg.get("architecture", "Unet")
    encoder = m_cfg.get("encoder_name", "resnet34")
    weights = m_cfg.get("encoder_weights", "imagenet")
    in_channels = m_cfg.get("in_channels", 1)
    classes = m_cfg.get("classes", 1)
    activation = m_cfg.get("activation", "sigmoid")

    if HAS_SMP:
        print(f"Building SMP model: {arch} with encoder {encoder} (weights={weights})...")
        model_cls = getattr(smp, arch, smp.Unet)
        model = model_cls(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    else:
        print("segmentation_models_pytorch not installed. Using pure PyTorch CustomUNet...")
        model = CustomUNet(in_channels=in_channels, classes=classes)

    return model
