import torch
from torchvision.datasets import ImageFolder
import numpy as np
cuda = torch.cuda.is_available()
from torchvision.transforms import ToTensor
import umap
import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, classes=None):
   

    plt.figure(figsize=(10,10))
    for i in range(len(classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.savefig("embeddings.png")

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 64)) #TODO 64 --> embedding_size
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def visualise():
    model = torch.load("model.pt")
    val_dataset = ImageFolder("data/teacher_splitted/Dishes/train", transform=ToTensor())
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    train_embeddings_tl, train_labels_tl = extract_embeddings(data_loader, model)
    dim_reduction = umap.UMAP()
    train_embeddings_low_dim = dim_reduction.fit_transform(train_embeddings_tl)
    plot_embeddings(train_embeddings_tl, train_labels_tl, classes=val_dataset.classes)





if __name__=="__main__":
    visualise()