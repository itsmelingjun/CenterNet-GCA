from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # 横向连接：统一通道到256
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # 顶层 P5: C5 -> 256
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        # 平滑层
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, c2, c3, c4, c5):
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # 平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


# -----------------------------------------------------------------#
#   使用Renset50作为主干特征提取网络，最终会获得一个
#   16x16x2048的有效特征层
# -----------------------------------------------------------------#
class ResNetFPN(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetFPN, self).__init__()
        # backbone
        # 512,512,3 -> 256,256,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 256x256x64 -> 128x128x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # change
        # 128x128x64 -> 128x128x256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 128x128x256 -> 64x64x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 64x64x512 -> 32x32x1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 32x32x1024 -> 16x16x2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # FPN
        self.fpn = FPN()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        return p2, p3, p4, p5


def resnet50FPN(pretrained=True):
    model = ResNetFPN(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='model_data/')
        # 注意：只加载与 backbone 匹配的部分
        backbone_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(backbone_dict, strict=False)
    return model


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class resnet50_SEDecoder(nn.Module):
    def __init__(self,inplanes=256 ,bn_momentum=0.1):
        super(resnet50_SEDecoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes  # fuse_conv输出通道
        self.deconv_with_bias = False
        # 注意力机制分配权重
        #self.se2 = SEBlock(256)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(256)
        self.se5 = SEBlock(256)
        # 融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256 * 3, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 上采样
        self.deconv_layers = self._make_deconv_layer(
            num_layers=1,
            num_filters=[64],
            num_kernels=[4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, p3, p4, p5):
        # SE增强
        #p2_se = self.se2(p2)
        p3_se = self.se3(p3)
        p4_se = self.se4(p4)
        p5_se = self.se5(p5)
        size = p3_se.shape[2:]
        #p3_up = F.interpolate(p3_se, size=size, mode='bilinear', align_corners=False) # 64*64 → 128*128
        p4_up = F.interpolate(p4_se, size=size, mode='bilinear', align_corners=False) # 32*32 → 128*128
        p5_up = F.interpolate(p5_se, size=size, mode='bilinear', align_corners=False) # 16*16 → 128*128
        # 拼接融合
        fused = torch.cat([p3_se, p4_up, p5_up], dim=1)
        fused = self.fuse_conv(fused)
        # 送入解码器上采样
        out = self.deconv_layers(fused)
        return out


class resnet50_maskHead(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(resnet50_maskHead, self).__init__()
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 512, 512, 1
        # -----------------------------------------------------------------#
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))
        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))
        # mask预测的部分
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        self.upsample_factor = 4

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        # 用heatmap引导mask分支
        att = x + x * hm.detach()
        mask = self.mask_head(att)
        # 上采样到原图大小
        mask = F.interpolate(mask, scale_factor=self.upsample_factor, mode='bilinear',
                             align_corners=False)  # (B, nc, H, W)

        return hm, wh, offset, mask
