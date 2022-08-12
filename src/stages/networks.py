import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ResNetModel
from torchvision.models.resnet import resnet50
from torchvision.models import resnet50, ResNet50_Weights
import torch
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 64, 7), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 3), nn.PReLU(),
                                     nn.MaxPool2d(2,stride=2))
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        # self.feature_extractor =  ResNetModel.from_pretrained("microsoft/resnet-50")
        self.resnet = resnet50()
        from torch.autograd import Variable
        modules=list(self.resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        weights = ResNet50_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.fc = nn.Sequential(nn.Linear(2048, 256),
                                nn.PReLU(),
                                # nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 64)
                                )

    def forward(self, x):
        # output = self.convnet(x)
        output = self.resnet(x)

        # output = output.last_hidden_state
        # print(output.size())
        output = output.view(output.size(0),-1)
        # x = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
