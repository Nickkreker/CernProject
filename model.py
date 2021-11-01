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
            nn.LeakyReLU(alpha, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels=16):
        super(Unet, self).__init__()


        self.conv1 = DoubleConvLayer(1, 64)
        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)


        self.conv2 = DoubleConvLayer(64, 128)
        self.dconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.conv3 = DoubleConvLayer(128, 256)
        self.dconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.conv4 = DoubleConvLayer(256, 512)
        self.uconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv5 = DoubleConvLayer(512, 256)
        self.uconv5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv6 = DoubleConvLayer(256, 128)
        self.uconv6 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv7 = DoubleConvLayer(128, 64)

        self.conv8 = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dconv1(x1)

        x2 = self.conv2(x2)
        x3 = self.dconv2(x2)

        x3 = self.conv3(x3)
        x4 = self.dconv3(x3)

        x4 = self.conv4(x4)
        x4 = self.uconv4(x4)

        x5 = self.conv5(torch.cat([x3, x4]), dim=1)
        x5 = self.uconv5(x5)

        x6 = self.conv6(torch.cat([x2, x5]), dim=1)
        x6 = self.uconv6(x6)

        x7 = self.conv7(torch.cat([x1, x6]), dim=1)

        x8 = self.conv8(x7)

        return x8




'''
class SUnet(nn.Module):
    def __init__(self):
        self(SUnet, self).__init__(channels=16)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3)
        self.unet1 = UnetLayer(16)


    def forward(self, x):
        pass
'''