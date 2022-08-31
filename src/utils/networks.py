import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ResNetModel
from torchvision.models.resnet import resnet50
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = transforms.functional.resize(x.float(),size=[224,224])
        x = x / 255.0

        x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=64):
        super(EmbeddingNet, self).__init__()
        # self.pretrained_model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.pretrained_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.transforms = Preprocess()
        modules=list(self.pretrained_model.children())[:-1]
        self.embedding=nn.Sequential(*modules)
        for p in self.pretrained_model.parameters():
            p.requires_grad = False



        self.embedding_size = embedding_size
        self.fc = nn.Sequential(nn.Linear(1024, 712),
                                nn.PReLU(),
                                nn.Linear(712, 256),
                                nn.PReLU(),
                                nn.Linear(256, self.embedding_size)
                                )

    def forward(self, x):
        output = self.transforms(x)
        output = self.embedding(output)
        output = output.view(output.size(0),-1)
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
