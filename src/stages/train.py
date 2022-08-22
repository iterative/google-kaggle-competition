import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

from src.utils.losses import batch_triplet_loss
from src.utils.networks import EmbeddingNet


def train_one_batch(model, input_data, labels, cuda):
    if cuda:
        input_data = input_data.cuda()
    hparams = "batch hard"
    batch_embeddings = model(input_data)
    loss = batch_triplet_loss(batch_embeddings, labels, hparams)
    return loss



def train(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    teacher_area = cli_params.teacher
    data_dir = Path(params["data"]["root"]) / params["data"]["train_data"] / teacher_area

    
    embedding_net = EmbeddingNet(embedding_size=params["train"]["embedding_size"])
    model = embedding_net
    train_dataset = ImageFolder(data_dir/"train", transform=embedding_net.transforms)
    val_dataset = ImageFolder(data_dir/"val", transform=embedding_net.transforms)

    cuda = torch.cuda.is_available()
    
    if cuda:
        model.cuda()

    # triplet_dataset_train = TripletMNIST(train_dataset, train=True) #, transform=model.embedding_net.transforms)
    # triplet_dataset_val = TripletMNIST(val_dataset, train=False) #, transform=model.embedding_net.transforms)


    
    kwargs = {'num_workers': 6, 'pin_memory': True} if cuda else {}
    batch_size = params["train"]["batch_size"]
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    margin = params["train"]["margin"]
    lr = params["train"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_epochs = params["train"]["epochs"]
    for epoch in range(2):
        epoch_loss = []
        for batch_idx, (data, labels) in enumerate(triplet_train_loader):
            loss = train_one_batch(model, data, labels, cuda)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Progress: {batch_idx}/{n_epochs} Loss: {np.mean(epoch_loss)}")

    # model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher
    # loss_fn = TripletMarginLoss(margin=margin)
    # loss_fn = batch_triplet_loss()

    # scheduler = lr_scheduler.StepLR(optimizer, 22, gamma=0.1, last_epoch=-1)
    # n_epochs = params["train"]["epochs"]
    # log_interval = 10
    #
    #
    # dvclive_path = Path(params["reports"]["root"]) / f"dvclive_{teacher_area}"
    # dvclive_path.mkdir(parents=True, exist_ok=True)
    # fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, log_folder=dvclive_path)
    #
    # model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher
    # model_path.mkdir(exist_ok=True)
    # torch.save(model, model_path/params["train"]["model_file"])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Dishes")
    parser.add_argument("--params", default="params.yaml")
    cli_params = parser.parse_args()
    train(cli_params)



