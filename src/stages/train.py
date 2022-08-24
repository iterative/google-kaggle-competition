import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import loguru
from src.utils.losses import batch_triplet_loss,OnlineTripletLoss
from src.utils.datasets import BalancedBatchSampler
from src.utils.triplet_selection_utils import SemihardNegativeTripletSelector
from src.utils.networks import EmbeddingNet
from dvclive import Live
from torch.optim import lr_scheduler
from src.utils.trainer import fit
from torchvision.transforms import ToTensor

def run_one_batch(model, input_data, labels, margin, cuda, hardest_only=True, test=False):
    if cuda:
        input_data = input_data.cuda()
    batch_embeddings = model(input_data)
    loss = batch_triplet_loss(batch_embeddings, labels, margin, hardest_only, test)
    return loss



def train(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    teacher_area = cli_params.teacher
    data_dir = Path(params["data"]["root"]) / params["data"]["train_data"] / teacher_area

    logger = loguru.logger
    embedding_net = EmbeddingNet(embedding_size=params["train"]["embedding_size"])
    model = embedding_net
    train_dataset = ImageFolder(data_dir/"train", transform=ToTensor())
    val_dataset = ImageFolder(data_dir/"val", transform=ToTensor())

    cuda = torch.cuda.is_available()
    
    if cuda:
        model.cuda()

    # triplet_dataset_train = TripletMNIST(train_dataset, train=True) #, transform=model.embedding_net.transforms)
    # triplet_dataset_val = TripletMNIST(val_dataset, train=False) #, transform=model.embedding_net.transforms)


    
    kwargs = {'num_workers': 6, 'pin_memory': True} if cuda else {}
    batch_size = params["train"]["batch_size"]
    n_classes = len(set(train_dataset.targets))
    n_samples = batch_size//n_classes
    train_batch_sampler = BalancedBatchSampler(np.array(train_dataset.targets), n_classes=n_classes, n_samples=n_samples)
    val_batch_sampler = BalancedBatchSampler(np.array(val_dataset.targets), n_classes=n_classes, n_samples=n_samples)
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset,batch_sampler=train_batch_sampler)#, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(val_dataset,batch_sampler=val_batch_sampler)#, batch_size=32, shuffle=False, **kwargs)

    margin = params["train"]["margin"]
    lr = params["train"]["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_epochs = params["train"]["epochs"]
    # loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
    scheduler = lr_scheduler.StepLR(optimizer, params["train"]["scheduler_step_size"], gamma=0.1, last_epoch=-1)
    dvclive_path = Path(params["reports"]["root"]) / f"dvclive_{teacher_area}"
    dvclive_path.mkdir(parents=True, exist_ok=True)
    # ================================================================================================================================
    live = Live(dvclive_path)
    hardest_only = False
    for epoch in range(n_epochs):
        train_epoch_loss = []
        model.train()
        if epoch==15:
            hardest_only = True
        for batch_idx, (data, labels) in enumerate(triplet_train_loader):

            loss = run_one_batch(model, data, labels, margin, cuda, hardest_only=hardest_only)
            # outputs = model(data.cuda())
            # loss,_ = loss_fn(outputs, labels)
            train_epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        live.log("learning_rate", scheduler.get_last_lr()[0])
        scheduler.step()
        train_loss = np.mean(train_epoch_loss)
        logger.info(f"Progress: {epoch}/{n_epochs} Loss: {train_loss}")
        live.log("training_loss", train_loss)


        val_epoch_loss = []
        model.eval()
        for batch_idx, (data, labels) in enumerate(triplet_test_loader):
            loss = run_one_batch(model, data, labels, margin, cuda, hardest_only=False, test=True)
            val_epoch_loss.append(loss.item())
        val_loss = np.mean(val_epoch_loss)
        live.log("validation_loss", val_loss)
        logger.info(f"Val loss: {val_loss}")

        live.next_step()
    #================================================================================================
        
    # model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher
    # # loss_fn = TripletMarginLoss(margin=margin)

    # scheduler = lr_scheduler.StepLR(optimizer, 22, gamma=0.1, last_epoch=-1)
    # n_epochs = params["train"]["epochs"]
    # log_interval = 10
    
    
    # dvclive_path = Path(params["reports"]["root"]) / f"dvclive_{teacher_area}"
    # dvclive_path.mkdir(parents=True, exist_ok=True)
    # fit(triplet_train_loader, triplet_test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, log_folder=dvclive_path)
    
    model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher
    model_path.mkdir(exist_ok=True)
    saved_model = torch.jit.script(model.embedding_net)
    saved_model.save(model_path/params["train"]["model_file"])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    cli_params = parser.parse_args()
    train(cli_params)



