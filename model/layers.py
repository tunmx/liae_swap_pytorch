import torch.nn as nn
from torch.nn import functional as F


class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Upscale, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels * 4, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pixel_shuffle(x, 2)
        return x

class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(Downscale, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            self.in_ch,
            self.out_ch,
            kernel_size=self.kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        return x

    def get_output_channels(self):
        return self.out_ch
    
class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, channels, n_downscales, kernel_size):
        super(DownscaleBlock, self).__init__()
        self.downs = []

        last_ch = in_channels
        for i in range(n_downscales):
            cur_ch = channels * (min(2 ** i, 8))
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size))
            last_ch = self.downs[-1].get_out_ch()

    def forward(self, inp):
        x = inp
        for down in self.downs:
            x = down(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=kernel_size // 2)
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pixel_shuffle(x, 2)
        
        identity = x
        x = self.residual(x)
        x = F.leaky_relu(identity + x, 0.2)
        
        return x
    
class SimpleUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SimpleUpscale, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, padding=kernel_size // 2)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pixel_shuffle(x, 2)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, inp):
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x
