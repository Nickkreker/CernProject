import torch
from torch import nn



'''
In double convolution authors use valid padding,
but they suggest usang 'same' because it much simplifies the description
of the network.

Instead of MaxPooling they use 'transpose' convolution (learned downsampling
filter).
'''
class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.03):
        super(DoubleConvLayer, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(alpha, inplace=True),
        )
#nn.BatchNorm2d(out_channels),
    def forward(self, x):
        return self.double_conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_dim=1, hidden_size=64):
        super(Unet, self).__init__()


        self.conv1 = DoubleConvLayer(in_channels, hidden_size)
        self.dconv1 = nn.MaxPool2d(2,)

        self.conv2 = DoubleConvLayer(hidden_size, 2 * hidden_size)
        self.dconv2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvLayer(2 * hidden_size, 4 * hidden_size)
        self.dconv3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvLayer(4 * hidden_size, 8 * hidden_size)
        self.uconv4 = nn.ConvTranspose2d(8 * hidden_size, 4 * hidden_size, kernel_size=2, stride=2)

        self.conv5 = DoubleConvLayer(8 * hidden_size, 4 * hidden_size)
        self.uconv5 = nn.ConvTranspose2d(4 * hidden_size, 2 * hidden_size, kernel_size=2, stride=2)

        self.conv6 = DoubleConvLayer(4 * hidden_size, 2 * hidden_size)
        self.uconv6 = nn.ConvTranspose2d(2 * hidden_size, hidden_size, kernel_size=2, stride=2)

        self.conv7 = DoubleConvLayer(2 * hidden_size, hidden_size)

        self.conv8 = nn.Conv2d(hidden_size, out_dim, kernel_size=1)


    def crop(self, x, target_shapes):
        _, _, height, width = x.shape
        diff_y = (height - target_shapes[0]) // 2
        diff_x = (width - target_shapes[0]) // 2
        return x[:, :, diff_y: (diff_y + target_shapes[0]), diff_x: (diff_x + target_shapes[1])]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dconv1(x1)

        x2 = self.conv2(x2)
        x3 = self.dconv2(x2)

        x3 = self.conv3(x3)
        x4 = self.dconv3(x3)

        x4 = self.conv4(x4)
        x4 = self.uconv4(x4)

        x5 = self.conv5(torch.cat([self.crop(x3, x4.shape[2:]), x4], dim=1))
        x5 = self.uconv5(x5)

        x6 = self.conv6(torch.cat([self.crop(x2, x5.shape[2:]), x5], dim=1))
        x6 = self.uconv6(x6)

        x7 = self.conv7(torch.cat([self.crop(x1, x6.shape[2:]), x6], dim=1))
        x8 = self.conv8(x7)

        return x8


class UnetEvo(nn.Module):
    def __init__(self, hidden_size=8):
        super(UnetEvo, self).__init__()

        self.unet1 = Unet(hidden_size=8)
        self.unet2 = Unet(hidden_size=4, in_channels=2)
        self.unet3 = Unet(hidden_size=4, in_channels=2)
        self.unet4 = Unet(hidden_size=4, in_channels=2)

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))

        x3 = self.unet3(torch.cat([x1, x2], dim=1))
        x4 = self.unet4(torch.cat([x2, x3], dim=1))

        return torch.cat([x1, x2, x3, x4], dim=1)

'''
class SUnet(nn.Module):
    def __init__(self):
        self(SUnet, self).__init__(channels=16)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3)
        self.unet1 = UnetLayer(16)


    def forward(self, x):
        pass
'''
