from genericpath import exists
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import yaml
from src.utils.datasets import TripletMNIST
from pathlib import Path
from src.utils.trainer import fit
import argparse
from torch.utils.data import random_split
from torch.nn import TripletMarginLoss
from src.utils.networks import EmbeddingNet, TripletNet
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


def train(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    data_dir = Path(params["data"]["root"]) / params["data"]["train"]

    
    embedding_net = EmbeddingNet(embedding_size=params["train"]["embedding_size"])
    model = TripletNet(embedding_net)
    train_dataset = ImageFolder(data_dir/"train", transform=ToTensor())
    val_dataset = ImageFolder(data_dir/"val", transform=ToTensor())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    triplet_dataset_train = TripletMNIST(train_dataset, train=True)
    triplet_dataset_val = TripletMNIST(val_dataset, train=False) 


    
    kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available()  else {}
    batch_size = params["train"]["batch_size"]
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

    margin = params["train"]["margin"]
    

    
    loss_fn = TripletMarginLoss(margin=margin)
    lr = params["train"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 22, gamma=0.1, last_epoch=-1)
    n_epochs = params["train"]["epochs"]
    log_interval = 10


    dvclive_path = Path(params["reports"]["root"]) / f"dvclive"
    dvclive_path.mkdir(parents=True, exist_ok=True)
    fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, log_folder=dvclive_path)
    
    model_path = Path(params["train"]["model_path"])
    model_path.mkdir(exist_ok=True)
    saved_model = torch.jit.script(model.embedding_net)
    saved_model.save(model_path/params["train"]["model_file"])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    cli_params = parser.parse_args()
    train(cli_params)



