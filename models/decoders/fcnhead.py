import torch.nn as nn


class FCNHead(nn.Module):

    def __init__(self, in_channels=[96, 192, 384, 768], channels=None,
                 kernel_size=3, dilation=1, num_classes=40,
                 norm_layer=nn.BatchNorm2d, in_index=-1, **kwargs):
        super(FCNHead, self).__init__()
        if isinstance(in_channels, (list, tuple)):
            sel_in_channels = in_channels[in_index]
        else:
            sel_in_channels = in_channels
        self.in_index = in_index
        self.kernel_size = kernel_size
        self.in_channels = sel_in_channels
        self.channels = channels or sel_in_channels // 4

        conv_padding = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels, kernel_size, padding=conv_padding),
                norm_layer(self.channels),
                nn.ReLU(inplace=True)
                )

        self.classifier = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[self.in_index]
        else:
            x = inputs
        output = self.conv(x)
        output = self.classifier(output)
        return output
