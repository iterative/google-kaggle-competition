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
    teacher_area = cli_params.teacher
    data_dir = Path(params["data"]["root"]) / params["data"]["train_data"] / teacher_area
    train_dataset = ImageFolder(data_dir/"train", transform=ToTensor())
    val_dataset = ImageFolder(data_dir/"val", transform=ToTensor())

    # test_size = 0.2
    # test_n_samples = int(len(dataset)*test_size)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])
    cuda = torch.cuda.is_available()




    # test_size = 0.2
    # test_n_samples = int(len(dataset)*test_size)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_n_samples, test_n_samples])
    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net)

    triplet_dataset_train = TripletMNIST(train_dataset, train=True, transform=model.embedding_net.transforms) 
    triplet_dataset_val = TripletMNIST(val_dataset, train=False, transform=model.embedding_net.transforms)
    # triplet_test_dataset = TripletMNIST(test_dataset)
    batch_size = 64
    kwargs = {'num_workers': 6, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

    # Set up the network and training parameters

    margin = 100
   

    if cuda:
        model.cuda()
    loss_fn = TripletMarginLoss(margin=margin)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 22, gamma=0.1, last_epoch=-1)
    
    n_epochs = 1
    log_interval = 10
    fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, log_folder=f"dvclive_{teacher_area}")#, metrics=[AccumulatedAccuracyMetric()])
    
    model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher 
    model_path.mkdir(exist_ok=True)
    torch.save(model, model_path/params["train"]["model_file"])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Dishes")
    parser.add_argument("--params", default="params.yaml")
    cli_params = parser.parse_args()
    train(cli_params)



