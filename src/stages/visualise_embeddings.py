import torch
from torchvision.datasets import ImageFolder
import numpy as np
import argparse
from torchvision.transforms import ToTensor
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import yaml 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, classes=None, save_path=None):
   

    plt.figure(figsize=(10,10))
    for i in range(len(classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    if save_path:
        plt.savefig(save_path)

def extract_embeddings(dataloader, model,embedding_size):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_size))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        cuda = torch.cuda.is_available()

        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def visualise(cli_params):
    params = yaml.safe_load(open(cli_params.params))

    model_path = Path(params["train"]["model_path"]) / "teacher" / cli_params.teacher / params["train"]["model_file"]
    model = torch.load(model_path)
    data_dir = Path(params["data"]["root"]) / params["data"]["train_data"] / cli_params.teacher
    val_dataset = ImageFolder(data_dir / "val", transform=model.embedding_net.transforms)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    train_embeddings_tl, train_labels_tl = extract_embeddings(data_loader, model, params["train"]["embedding_size"])
    dim_reduction = umap.UMAP()
    train_embeddings_low_dim = dim_reduction.fit_transform(train_embeddings_tl)
    output_path = Path(params["reports"]["root"]) / Path(params["reports"]["plots"]) / cli_params.teacher 
    output_path.mkdir(parents=True, exist_ok=True)
    plot_embeddings(train_embeddings_low_dim, train_labels_tl, classes=val_dataset.classes, save_path=output_path / "embedding.png")





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Dishes")
    parser.add_argument("--params", default="params.yaml")
    cli_params = parser.parse_args()
    visualise(cli_params)