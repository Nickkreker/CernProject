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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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

        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        x5 = self.uconv5(x5)

        x6 = self.conv6(torch.cat([x2, x5], dim=1))
        x6 = self.uconv6(x6)

        x7 = self.conv7(torch.cat([self.crop(x1, x6.shape[2:]), x6], dim=1))
        x8 = self.conv8(x7)

        return x8


class UnetEvo(nn.Module):
    def __init__(self, hidden_size=8):
        super(UnetEvo, self).__init__()

        self.unet1 = Unet(hidden_size=32, out_dim=4)
        self.unet2 = Unet(hidden_size=64, in_channels=6, out_dim=5)

    def forward(self, x):
        x1 = self.unet1(x)
        t = torch.ones_like(x)
        t *= 0.18
        x2 = self.unet2(torch.cat([x, t, x1], dim=1))

        return torch.cat([x1, x2], dim=1)

class UnetEvoMod(nn.Module):
    def __init__(self, hidden_size=8):
        super(UnetEvoMod, self).__init__()

        self.unet1 = Unet(hidden_size=32, out_dim=4)
        self.unet2 = Unet(hidden_size=64, in_channels=6, out_dim=4)
        self.unet3 = Unet(hidden_size=32, in_channels=9, out_dim=1)

    def forward(self, x):
        x1 = self.unet1(x)
        t = torch.ones_like(x)
        t *= 0.18
        x2 = self.unet2(torch.cat([x, t, x1], dim=1))
        x3 = self.unet3(torch.cat([x, x1, x2], dim=1))

        return torch.cat([x1, x2, x3], dim=1)

class MultiPathUnet(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, out_dim=1):
        super(MultiPathUnet, self).__init__()

        self.conv1 = DoubleConvLayer(in_channels, hidden_size)
        self.dconv1 = nn.MaxPool2d(2,)

        self.conv2 = DoubleConvLayer(hidden_size, 2 * hidden_size)
        self.dconv2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvLayer(2 * hidden_size, 4 * hidden_size)
        self.dconv3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvLayer(12 * hidden_size, 8 * hidden_size)
        self.uconv4 = nn.ConvTranspose2d(8 * hidden_size, 4 * hidden_size, kernel_size=2, stride=2)

        self.conv5 = DoubleConvLayer(16 * hidden_size, 4 * hidden_size)
        self.uconv5 = nn.ConvTranspose2d(4 * hidden_size, 2 * hidden_size, kernel_size=2, stride=2)

        self.conv6 = DoubleConvLayer(8 * hidden_size, 2 * hidden_size)
        self.uconv6 = nn.ConvTranspose2d(2 * hidden_size, hidden_size, kernel_size=2, stride=2)

        self.conv7 = DoubleConvLayer(4 * hidden_size, hidden_size)

        self.conv8 = nn.Conv2d(hidden_size, out_dim, kernel_size=1)

    def _forward_down(self, x):
        x1 = self.conv1(x)
        x2 = self.dconv1(x1)

        x2 = self.conv2(x2)
        x3 = self.dconv2(x2)

        x3 = self.conv3(x3)
        x4 = self.dconv3(x3)
        
        return x1, x2, x3, x4


    def forward(self, x):
        # ed = x[0], velx = x[1], vely = x[2]
        ed1, ed2, ed3, ed4 = self._forward_down(torch.unsqueeze(x[:,0], 1))
        velx1, velx2, velx3, velx4 = self._forward_down(torch.unsqueeze(x[:,1], 1))
        vely1, vely2, vely3, vely4 = self._forward_down(torch.unsqueeze(x[:,2], 1))

        x4 = self.conv4(torch.cat([ed4, velx4, vely4], dim=1))
        x4 = self.uconv4(x4)

        x5 = self.conv5(torch.cat([ed3, velx3, vely3, x4], dim=1))
        x5 = self.uconv5(x5)

        x6 = self.conv6(torch.cat([ed2, velx2, vely2, x5], dim=1))
        x6 = self.uconv6(x6)

        x7 = self.conv7(torch.cat([ed1, velx1, vely1, x6], dim=1))
        x8 = self.conv8(x7)

        return x8

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = DoubleConvLayer(1, 128)
        self.conv2 = DoubleConvLayer(128, 256)
        self.conv3 = DoubleConvLayer(256, 64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class MIConvNet(nn.Module):
    def __init__(self):
        super(MIConvNet, self).__init__()
        self.conv1 = DoubleConvLayer(1, 60)
        self.conv2 = DoubleConvLayer(60, 200)
        self.conv3 = DoubleConvLayer(200, 100)
        self.conv4 = nn.Conv2d(100, 1, kernel_size=1)

    def forward(self, x):
        ed = self.conv1(torch.unsqueeze(x[:,0], 1))
        vx = self.conv1(torch.unsqueeze(x[:,1], 1))
        vy = self.conv1(torch.unsqueeze(x[:,2], 1))
        x = self.conv2(torch.maximum(torch.maximum(ed, vx), vy))
        x = self.conv3(x)
        x = self.conv4(x)
        return x



'''
class SUnet(nn.Module):
    def __init__(self):
        super(SUnet, self).__init__()
        self.unet1 = Unet(hidden_size=64)
        self.unet2 = Unet(in_channels=64)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        x = self.unet1(x)
        x = self.unet2(x)
        return self.conv(x)
'''