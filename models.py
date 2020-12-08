import torch
from torch import nn



class ResidualBlock(nn.Module):
    """
    ResidualBlock Class
        Performs two convolutions and an instance normnlization, the input is added
        to this output to form the residual block output.

    Args:
        input_channels (int): the number of channels to expect from a given input
    """
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Method for completing a forward pass of ResidualBlock

        Args:
            x (tensor): Image tensor of shape (batch_size, num_channels, heioght, width)
        """
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x


class ContractingBlock(nn.Module):
    """
    Contracting Block Class
    Performs a convolution followed by a max pool operation and an optional instance norm.

    Args:
        input_channels (int): The number of channels to expect from a given input
    """
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode="reflect")
        self.activation = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        """Method for completing a forward pass of COntactingBlock

        Args:
            x (tensor): Image tensor of shape (batch_size, n_channels, height, width)
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    """ExpandingBlock Class
        Performs a convolutional transpose operation ir order to upsample with optional instance norm

    Args:
        input_channels (int): The number of channels to expect from a gicen input
    """
    def __init__(self, input_channels, use_bn=True):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels //2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:  
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        """Method for completing a forward pass of ExpandingBlock

        Args:
            x (tensor): Image tensor of shape (batch_size, n_channels, height, width)
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    """FeatureMapBlock Class
        The final layer of the generator
    Args:
        input_channels (int): the number of channels to expect from a given input
        output_channels (int): the number of channels to expect for a given output
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode="reflect")
    
    def forward(self, x):
        """Method for completing a forward pass of FeatureMap Block

        Args:
            x (tensor): Image tensor of shape (batch_size, n_channels, height, width)
        """
        x = self.conv(x)
        return x


class Generator(nn.Module):
    """Generator Class
        2 Conctracting Blocks, 9 Residual Blocks, 2 Expanding Blocks
    Args:
        input_channels (int): number of channels to expect from a given input
        output_channels (int): number of channels to expect from a given output
    """
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2 )
        res_mult = 4  
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand1 = ExpandingBlock(hidden_channels * 4)
        self.expand2 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Method for completing a forward pass of the Generator

        Args:
            x (tensor): Image tensor of shape (batch_size, n_channels, height, width)
        """
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand1(x11)
        x13 = self.expand2(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)


class Dicriminator(nn.Module):
    """Discriminator Class

    Args:
        input_channels (int): the number of image input channels
        hidden_channels (int): the initial number of discriminator convolutional filters
    """
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation="lrelu")
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation="lrelu")
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation="lrelu")
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size= 1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn
        
