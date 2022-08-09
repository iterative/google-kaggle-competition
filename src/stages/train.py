import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from datasets import TripletMNIST

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from networks import EmbeddingNet, TripletNet
from torchvision.datasets import ImageFolder
from losses import TripletLoss
from metrics import AccumulatedAccuracyMetric



def train():
    dataset = ImageFolder("data/teacher_data/Dishes")

    test_size = 0.2
    test_n_samples = int(len(dataset)*test_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])
    cuda = torch.cuda.is_available()

    from torch.utils.data import random_split



    test_size = 0.2
    test_n_samples = int(len(dataset)*test_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])

    triplet_dataset_train = TripletMNIST(train_dataset.dataset, train=True, subset_indices=train_dataset.indices) 
    triplet_dataset_val = TripletMNIST(val_dataset.dataset, train=False, subset_indices=val_dataset.indices)
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
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 4, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 10
    fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)#, metrics=[AccumulatedAccuracyMetric()])

    torch.save(model, "model.pt")

if __name__=="__main__":
    train()



