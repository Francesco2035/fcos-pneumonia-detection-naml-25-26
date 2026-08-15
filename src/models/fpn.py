import torch, torchvision, torch.nn as nn
from src.models.backbone import Backbone

class FPN(nn.Module):
    def __init__(self, path_model, device):
        super().__init__()
        self.backbone = Backbone(path_model, device)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        # lateral connections : faccio tutti a 256 channel
        self.lat_c2 = nn.Conv2d(256, 256, 1)
        self.lat_c3 = nn.Conv2d(512, 256, 1)
        self.lat_c4 = nn.Conv2d(1024, 256, 1)
        self.lat_c5 = nn.Conv2d(2048, 256, 1)

        # 3x3 convolutions : per pulire la somma
        self.conv_p2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_p3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_p4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_p5 = nn.Conv2d(256, 256, 3, padding=1)     


    def forward(self, x):
        C2, C3, C4, C5 = self.backbone(x)

        # P5
        P5 = self.lat_c5(C5)

        # P4
        P5_up = self.upsampling(P5)

        C4_lat = self.lat_c4(C4)

        P4_merged = C4_lat + P5_up

        P4 = self.conv_p4(P4_merged)

        # P3
        P4_up = self.upsampling(P4)

        C3_lat = self.lat_c3(C3)

        P3_merged = C3_lat + P4_up

        P3 = self.conv_p3(P3_merged)

        # P2
        P3_up = self.upsampling(P3)

        C2_lat = self.lat_c2(C2)

        P2_merged = C2_lat + P3_up

        P2 = self.conv_p2(P2_merged)

        return P2, P3, P4, P5

