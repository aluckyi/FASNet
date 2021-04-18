import torch.nn as nn
import torch

import torch.nn.functional as F
from .module import MSFA
from .enet import backbone_net


# add MSFA + FSLoss
class FASNet(nn.Module):
    def __init__(self, num_classes=3):
        super(FASNet, self).__init__()

        self.backbone = backbone_net()
        self.msfa = MSFA(in_channels=416, mid_channels=64, out_channels=16)

        self.cls_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, inputs):
        # Initial block
        x0 = self.backbone.initial_block(inputs)

        # Stage 1 - Encoder
        stage1_input_size = x0.size()
        x, max_indices1_0 = self.backbone.downsample1_0(x0)
        x = self.backbone.regular1_1(x)
        x = self.backbone.regular1_2(x)
        x = self.backbone.regular1_3(x)
        x1 = self.backbone.regular1_4(x)

        # Stage 2 - Encoder
        stage2_input_size = x1.size()
        x, max_indices2_0 = self.backbone.downsample2_0(x1)
        x = self.backbone.regular2_1(x)
        x = self.backbone.dilated2_2(x)
        x = self.backbone.asymmetric2_3(x)
        x = self.backbone.dilated2_4(x)
        x = self.backbone.regular2_5(x)
        x = self.backbone.dilated2_6(x)
        x = self.backbone.asymmetric2_7(x)
        x2 = self.backbone.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.backbone.regular3_0(x2)
        x = self.backbone.dilated3_1(x)
        x = self.backbone.asymmetric3_2(x)
        x = self.backbone.dilated3_3(x)
        x = self.backbone.regular3_4(x)
        x = self.backbone.dilated3_5(x)
        x = self.backbone.asymmetric3_6(x)
        x3 = self.backbone.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.backbone.upsample4_0(x3, max_indices2_0, output_size=stage2_input_size)
        x = self.backbone.regular4_1(x)
        x4 = self.backbone.regular4_2(x)

        # Stage 5 - Decoder
        x = self.backbone.upsample5_0(x4, max_indices1_0, output_size=stage1_input_size)
        x5 = self.backbone.regular5_1(x)

        # feature fusion
        size = x5.size()[-2:]
        x_1 = F.interpolate(x1, size=size, mode='bilinear', align_corners=True)
        x_2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=True)
        x_3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=True)
        x_4 = F.interpolate(x4, size=size, mode='bilinear', align_corners=True)

        y = torch.cat([x0, x_1, x_2, x_3, x_4, x5], dim=1)
        y = self.msfa(y)

        x = self.cls_conv(x5+y)
        # # =============================================
        # import os
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from utils import draw_features
        # feat = y.detach().cpu().numpy()
        # savename = os.path.join('./heatmap', 'heatmap_y.png')
        # draw_features(4, 4, feat, savename)
        # # =============================================

        pred = F.interpolate(x, size=inputs.size()[-2:], mode='bilinear', align_corners=True)

        return pred, y
