import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet, ResNet18_Weights

from horizon.base.model import BaseModel


def resnet18_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class TwoStreamNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        ResNet.forward = resnet18_forward
        # body steam
        self.body_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        # scene-stream
        model_weights = torch.hub.load_state_dict_from_url(
            "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
        )
        model_weights = {
            str.replace(k, 'module.', ''): v
            for k, v in model_weights['state_dict'].items()
        }
        self.scene_extractor = models.__dict__["resnet18"](num_classes=365)
        self.scene_extractor.load_state_dict(model_weights)

    def forward(self, body, image):
        I_b = self.body_extractor(body)
        I_s = self.scene_extractor(image)
        return I_b, I_s


class FusionNetwork(nn.Module):

    def __init__(self, body_dim, scene_dim):
        super().__init__()

        self.fc1 = nn.Linear((body_dim + scene_dim), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, body, scene):
        fuse_features = torch.cat((body, scene), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return {'cat': cat_out, 'cont': cont_out}


class EMOTIC(BaseModel):

    def __init__(self):
        super().__init__()
        self.two_stream_module = TwoStreamNetwork()
        self.fusion_module = FusionNetwork(body_dim=512, scene_dim=512)

    def forward(self, body, image):
        body, scene = self.two_stream_module(body, image)
        return self.fusion_module(body, scene)
