import torch
from torch import nn
import torch.nn.functional as F


def BCE_KLD_loss(input, target, mu, logvar):
    BCE = F.binary_cross_entropy(input, target)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input.numel()
    return BCE + KLD

    
def MSE_KLD_loss(input, target, mu, logvar):
    MSE = torch.mean((input-target).pow(2))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input.numel()
    return MSE + KLD


class Encoder(nn.Module):
    def __init__(self, debug):
        super(Encoder, self).__init__()

        self.debug = debug

        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 4, 3, 1, 0)
        self.conv3 = nn.Conv2d(4, 8, 3, 1, 0)
        self.conv41 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv42 = nn.Conv2d(8, 16, 3, 1, 1)


    def print_debug(self, x):
        if self.debug:
            print(x.shape)

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = F.relu(out)

        self.print_debug(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)

        self.print_debug(out)

        out = self.conv3(out)
        out = self.pool3(out)
        out = F.relu(out)

        self.print_debug(out)

        mean = self.conv41(out)
        mean = self.pool41(mean)
        mean = F.relu(mean)

        self.print_debug(mean)

        std = self.conv42(out)
        std = self.pool42(std)
        std = F.relu(std)

        self.print_debug(std)

        return mean.view(mean.shape[0], -1), std.view(std.shape[0], -1)


class MaxPoolEncoder(Encoder):
    def __init__(self, debug=False):
        super(MaxPoolEncoder, self).__init__(debug)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool41 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool42 = nn.MaxPool2d(kernel_size=2, stride=2)


class ConvPoolEncoder(Encoder):
    def __init__(self, debug=False):
        super(ConvPoolEncoder, self).__init__(debug)

        self.pool1 = nn.Conv2d(2, 2, 2, 2, 0)
        self.pool2 = nn.Conv2d(4, 4, 2, 2, 0)
        self.pool3 = nn.Conv2d(8, 8, 2, 2, 0)
        self.pool41 = nn.Conv2d(16, 16, 2, 2, 0)
        self.pool42 = nn.Conv2d(16, 16, 2, 2, 0)


class Decoder(nn.Module):
    def __init__(self, debug=False):
        super(Decoder, self).__init__()
        
        self.debug = debug
        self.transconv1 = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        self.transconv2 = nn.ConvTranspose2d(8, 4, 4, 2, 0)
        self.transconv3 = nn.ConvTranspose2d(4, 2, 4, 2, 0)
        self.transconv4 = nn.ConvTranspose2d(2, 1, 4, 2, 1)
    

    def print_debug(self, x):
        if self.debug:
            print(x.shape)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        self.print_debug(x)

        out = self.transconv1(x)
        out = F.relu(out)
        self.print_debug(out)

        out = self.transconv2(out)
        out = F.relu(out)
        self.print_debug(out)

        out = self.transconv3(out)
        out = F.relu(out)
        self.print_debug(out)

        out = self.transconv4(out)
        out = torch.sigmoid(out)
        self.print_debug(out)

        return out


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
