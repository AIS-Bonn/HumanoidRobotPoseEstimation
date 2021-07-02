from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def convtrans3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def upsample(scale):
    return nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)


class PoseNet(nn.Module):

    def __init__(self, outputs, final_kernel_size, backbone, pretrained=True):
        super(PoseNet, self).__init__()
        self.outputs = outputs
        if backbone == 'resnet18':
            inplanes = [512, 256, 128, 64, 64]
            planes = 128
            self.encoder = resnet18(pretrained=pretrained, progress=False, replace_stride_with_dilation=[False, False, False], norm_layer=None)
            del self.encoder.fc
            del self.encoder.avgpool
        else:
            raise ValueError(f'Backbone {backbone} not supported')

        # lateral/skip connections
        self.conn = nn.Sequential(OrderedDict([
            ('lateral1', conv1x1(inplanes[1], planes)),
            ('lateral2', conv1x1(inplanes[2], planes)),
            ('lateral3', conv1x1(inplanes[3], planes)),
            ('lateral4', conv1x1(inplanes[4], planes))
        ]))

        final_conv = conv1x1 if final_kernel_size == 1 else conv3x3
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                convtrans3x3(inplanes[0], planes),
                upsample(2)
            )),
            ('layer2', nn.Sequential(
                nn.BatchNorm2d(2 * planes),
                nn.ReLU(inplace=True),
                convtrans3x3(2 * planes, planes),
                upsample(2)
            )),
            ('layer3', nn.Sequential(
                nn.BatchNorm2d(2 * planes),
                nn.ReLU(inplace=True),
                convtrans3x3(2 * planes, planes),
                upsample(2)
            )),
            ('layer4', nn.Sequential(
                nn.BatchNorm2d(2 * planes),
                nn.ReLU(inplace=True)
            )),
            ('layer5', nn.Sequential(
                convtrans3x3(2 * planes + sum(outputs), planes),
                upsample(2)
            )),
            ('layer6', nn.Sequential(
                nn.BatchNorm2d(2 * planes),
                nn.ReLU(inplace=True)
            )),
            ('lastconv1', final_conv(2 * planes, sum(outputs), bias=True)),
            ('lastconv2', final_conv(2 * planes, outputs[1], bias=True))
        ]))

        self._init_parameters()
    
    def _init_parameters(self):
        for m in self.conn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x1 = self.encoder.relu(x)
        x = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        return x1, x2, x3, x4, x5

    def forward(self, x):
        c4, c3, c2, c1, x = self.forward_encoder(x)

        x = self.decoder.layer1(x)
        c1 = self.conn.lateral1(c1)
        x = torch.cat([x, c1], dim=1)

        x = self.decoder.layer2(x)
        c2 = self.conn.lateral2(c2)
        x = torch.cat([x, c2], dim=1)

        x = self.decoder.layer3(x)
        c3 = self.conn.lateral3(c3)
        x = torch.cat([x, c3], dim=1)

        x = self.decoder.layer4(x)
        out1 = self.decoder.lastconv1(x)
        cx = torch.cat([out1, x], dim=1)

        x = self.decoder.layer5(cx)
        c4 = self.conn.lateral4(c4)
        x = torch.cat([x, c4], dim=1)
        x = self.decoder.layer6(x)

        out2 = self.decoder.lastconv2(x)

        limbs, hms = torch.split(out1.unsqueeze(1), self.outputs, dim=2)
        losses = [hms, out2.unsqueeze(1), limbs, None]

        return [hms[:, -1], limbs[:, -1]] + losses
