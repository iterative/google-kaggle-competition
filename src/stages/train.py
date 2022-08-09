import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from datasets import TripletMNIST

from trainer import fit
import numpy as np
import argparse 
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from networks import EmbeddingNet, TripletNet
from torchvision.datasets import ImageFolder
from losses import TripletLoss
from metrics import AccumulatedAccuracyMetric
from torchvision.transforms import ToTensor


def train():
    train_dataset = ImageFolder("data/teacher_splitted/Dishes/train", transform=ToTensor())
    val_dataset = ImageFolder("data/teacher_splitted/Dishes/train", transform=ToTensor())

    # test_size = 0.2
    # test_n_samples = int(len(dataset)*test_size)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])
    cuda = torch.cuda.is_available()




    # test_size = 0.2
    # test_n_samples = int(len(dataset)*test_size)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])

    triplet_dataset_train = TripletMNIST(train_dataset, train=True) 
    triplet_dataset_val = TripletMNIST(val_dataset, train=False)
    # triplet_test_dataset = TripletMNIST(test_dataset)
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

    # Set up the network and training parameters

    margin = 1.
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)

    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.1, last_epoch=-1)
    
    n_epochs = 4
    log_interval = 10
    fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)#, metrics=[AccumulatedAccuracyMetric()])

    torch.save(model, "model.pt")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Dishes")
    cli_params = parser.parse_args()
    train()


