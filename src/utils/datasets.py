import json
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset, train, subset_indices=None, transform=None):

        # if subset_indices is not None:
        #     self.dataset = dataset[subset_indices]
        # else: 
        #     self.dataset = dataset
        self.dataset = dataset
        self.train = train
        self.transform = transform
        self.labels = np.array(self.dataset.targets)
        # self.data = self.dataset
        self.subset_indices = subset_indices
        self.labels_set = set(self.dataset.targets)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

        # self.label_to_indices = {label: [] for label in self.labels_set}
        # for index in subset_indices:
        #     self.label_to_indices[self.labels[index]].append(index)


        random_state = np.random.RandomState(29)
        if not train: 
            triplets = [[i,
                            random_state.choice(self.label_to_indices[self.labels[i].item()]),
                            random_state.choice(self.label_to_indices[
                                                    np.random.choice(
                                                        list(self.labels_set - set([self.labels[i].item()]))
                                                    )
                                                ])
                             ]
                            for i in range(len(self.dataset))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.dataset[index], self.labels[index] #.item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.dataset[positive_index]
            img3 = self.dataset[negative_index]
        else:
            img1 = self.dataset[self.test_triplets[index][0]]
            img2 = self.dataset[self.test_triplets[index][1]]
            img3 = self.dataset[self.test_triplets[index][2]]
        img1 = img1[0]
        img2 = img2[0]
        img3 = img3[0]

        # img1 = pil_to_tensor(img1[0])/255
        # img2 = pil_to_tensor(img2[0])/255
        # img3 = pil_to_tensor(img3[0])/255
        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class FiftyOneTorchDataset(Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        index_file_folder: Path to the folder with labels.json and
            either 'data' folder or 'manifest.json' file.
        manifest_path: Path to the manifest.json for the FiftyOne dataset.
            If left None and there is no 'manifest.json' file in 'index_file_folder',
            then it is assumed that images 'index_file_folder/data'.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset 
            that contains the desired labels to load
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(
        self,
        index_file_folder,
        manifest_path=None,
        transform=None,
        gt_field="ground_truth",
        target_transform=None,
    ):
        self.index_file_folder = Path(index_file_folder)
        self.labels_path = self.index_file_folder/'labels.json'
        self.manifest_path = manifest_path
        self.transform = transform
        self.gt_field = gt_field
        self.target_transform = target_transform

        with open(self.labels_path) as json_file:
            self.labels = json.load(json_file)

        if (index_file_folder/'manifest.json').is_file():
            self.manifest_path = index_file_folder/'manifest.json'

        if self.manifest_path is None:
            files = [p for p in (index_file_folder/'data').iterdir() if p.is_file()]
            self.manifest = {file.name.split('.')[0]:file.resolve().as_posix() for file in files}
        else:
            with open(self.manifest_path) as json_file:
                self.manifest = json.load(json_file)

        self.img_paths = [self.manifest[img_name] for img_name in self.labels['labels'].keys()]

        self.classes = np.unique(list(self.labels['labels'].values())).tolist()

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[lbl] for lbl in self.labels['labels'].values()]
        self.samples = [(self.manifest[img_name],self.class_to_idx[lbl]) for img_name, lbl in self.labels['labels'].items()]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_classes(self) -> Tuple[Any]:
        return self.classes