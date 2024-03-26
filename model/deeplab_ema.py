import torch
import torch.nn as nn
import torch.nn.functional as F
from model.batchnorm import SynchronizedBatchNorm2d
from model.basicnet import build_aspp, build_deeplab_decoder 
from model.resnet import ResNet101
from model.mobilenet import MobileNetV2


class DeepLab_EMA(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_EMA, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = MobileNetV2(output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_deeplab_decoder(num_classes, backbone, BatchNorm)
        self.dis = nn.Sequential(BatchNorm(305), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(305, 1, kernel_size=1, stride=1))
        #self.dis_b = nn.Sequential(BatchNorm(304), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(304, 1, kernel_size=1, stride=1))
        self.reshape_channel = nn.Conv2d(320, 3, 1, stride=1, padding=0, bias=False)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat, domain_info = self.backbone(input)
        domain_info = self.reshape_channel(domain_info)
        domain_info = F.interpolate(domain_info, scale_factor=16, mode="bilinear")

        return domain_info

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield 