import argparse
from pathlib import Path
from pkgutil import get_data
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import loguru
from src.utils.losses import batch_triplet_loss
from src.utils.datasets import BalancedBatchSampler
from src.utils.triplet_selection_utils import SemihardNegativeTripletSelector
from src.utils.networks import EmbeddingNet
from dvclive import Live
from torch.optim import lr_scheduler
from src.utils.trainer import fit
from torchvision.transforms import ToTensor
from pytorch_metric_learning import miners, losses


def run_epoch(data_loader, model, loss_func, miner, optimizer, margin, device, is_train=True):
        epoch_loss = []
        model = model.train() if is_train else model.eval()

        for batch_idx, (data, labels) in enumerate(data_loader):
            loss = run_one_batch(model, data, labels, loss_func, miner, margin, device, hardest_only=False)
            epoch_loss.append(loss.item())
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return np.mean(epoch_loss)


def run_one_batch(model, input_data, labels,loss_func, miner, margin, device, hardest_only=True, test=False):
    
    input_data = input_data.to(device)
    input_data = input_data.half()
    labels = labels.to(device)
    labels = labels
    batch_embeddings = model(input_data)
    # hard_pairs = miner(batch_embeddings, labels)
    # loss = loss_func(batch_embeddings, labels, hard_pairs)
    loss = batch_triplet_loss(batch_embeddings, labels, margin, hardest_only)
    return loss

def get_data_loader(dataset, batch_size):
        n_classes = len(set(dataset.targets)) 
        n_samples = batch_size//n_classes
        kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
        batch_sampler = BalancedBatchSampler(np.array(dataset.targets), n_classes=n_classes, n_samples=n_samples)
        return torch.utils.data.DataLoader(dataset,batch_sampler=batch_sampler, **kwargs)

def train(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = cli_params.dataset
    data_dir = Path(params["data"]["root"]) / dataset_name
    
    model = EmbeddingNet(embedding_size=params["train"]["embedding_size"])
    model.to(device)    

    train_dataset = FiftyOneTorchDataset(Path(params['data_preprocessing']['index_file_folder'])/'train', transform=ToTensor())
    val_dataset = FiftyOneTorchDataset(Path(params['data_preprocessing']['index_file_folder'])/'val', transform=ToTensor())

    batch_size = params["train"]["batch_size"]
    margin = params["train"]["margin"]
    lr = params["train"]["lr"]
    n_epochs = params["train"]["epochs"]

    triplet_train_loader = get_data_loader(train_dataset, batch_size=batch_size)
    triplet_test_loader = get_data_loader(val_dataset, batch_size=batch_size)
    
    miner = miners.TripletMarginMiner(type_of_triplets="semihard")
    loss_func = losses.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, params["train"]["scheduler_step_size"], gamma=0.1, last_epoch=-1)

    dvclive_path = Path(params["reports"]["root"]) / f"dvclive_{dataset_name}"
    dvclive_path.mkdir(parents=True, exist_ok=True)
    live = Live(dvclive_path)

    logger = loguru.logger
    for epoch in range(n_epochs):

        train_loss = run_epoch(triplet_train_loader, model, loss_func, miner, margin=margin, device=device, optimizer=optimizer, is_train=True)
        live.log("learning_rate", scheduler.get_last_lr()[0])
        scheduler.step()
        logger.info(f"Progress: {epoch}/{n_epochs} Loss: {train_loss}")
        live.log("training_loss", train_loss)

        val_loss = run_epoch(triplet_test_loader, model,loss_func, miner, margin=margin, device=device, optimizer=optimizer, is_train=False)
        live.log("validation_loss", val_loss)
        logger.info(f"Val loss: {val_loss}")
        live.next_step()
    
    model_path = Path(params["train"]["model_path"]) / dataset_name
    model_path.mkdir(exist_ok=True)
    saved_model = torch.jit.script(model)
    saved_model.save(model_path/params["train"]["model_file"])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset", default="baseline_130k_128_split")
    cli_params = parser.parse_args()
    train(cli_params)



