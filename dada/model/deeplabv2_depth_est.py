import torch
import torch.nn as nn

from advent.model.deeplabv2 import Bottleneck, ResNetMulti, ClassifierModule

AFFINE_PAR = True


class ResNetMultiDepth(ResNetMulti):
    def __init__(self, block, layers, num_classes, multi_level):
        super().__init__(block, layers, num_classes, multi_level)

        self.conv_depth = ClassifierModule(2048, [6, 12], [6, 12], 2048)
        self.conv_seg = ClassifierModule(2048, [6, 12], [6, 12], 2048)
        self.conv_out_seg = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.conv_out_seg_w_depth = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.conv_out_depth = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            seg_conv4 = self.layer5(x)  # produce segmap 1, not used in DADA
        else:
            seg_conv4 = None
        x4 = self.layer4(x)
        # encoder
        x4_depth = self.conv_depth(x4)
        x4_seg = self.conv_seg(x4)
        out_depth =self.conv_out_depth(x4_depth)
        out_seg = self.conv_out_seg(x4_seg)
        out_seg_w_depth = self.conv_out_seg_w_depth(x4_depth)

        return seg_conv4, out_seg, out_depth, out_seg_w_depth

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())    
        b.append(self.conv_seg.parameters())
        b.append(self.conv_depth.parameters())
        b.append(self.conv_out_seg_w_depth.parameters())
        b.append(self.conv_out_depth.parameters())
        b.append(self.conv_out_seg.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


def get_deeplab_v2_depth_est(num_classes=16, multi_level=False):
    model = ResNetMultiDepth(
        Bottleneck, [3, 4, 23, 3], num_classes, multi_level
    )
    return model
