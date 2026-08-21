import torch
import torch.nn as nn
import torchvision

from src.models.resnet import ResNet50


class Backbone(nn.Module):

    def __init__(self, path_model):
        super().__init__()

        if path_model is None:

            self.model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )

        else:

            self.model = ResNet50(3, 2)

            checkpoint = torch.load(
                path_model,
                map_location="cpu"
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        C2 = self.model.layer1(x)
        C3 = self.model.layer2(C2)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        return C2, C3, C4, C5