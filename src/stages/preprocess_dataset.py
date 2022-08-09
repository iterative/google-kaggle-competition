import argparse
from pathlib import Path
import torch
from torchvision import models , transforms, datasets
import yaml


# TODO: This may be revised. Not sure what transformations are needed.
def main(params):
    stage_params = yaml.safe_load(open(params))
    data_root = Path(stage_params["unzip_dataset"]['data_root'])
    input_folder = data_root/Path(stage_params["unzip_dataset"]['dataset_baseline'])
    output_folder = data_root/Path(stage_params["preprocess_dataset"]['output_folder'])

    batch_size = stage_params["preprocess_dataset"]["batch_size"]
    center_crop = stage_params["preprocess_dataset"]["center_crop"]
    normalize_mean = stage_params["preprocess_dataset"]["normalize_mean"]
    normalize_std = stage_params["preprocess_dataset"]["normalize_std"]
    resize = stage_params["preprocess_dataset"]["resize"]

    transform = transforms.Compose([transforms.Resize(resize),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)])

    dataset = datasets.ImageFolder(input_folder, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch.save(dataloader, output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='params.yaml')
    args = parser.parse_args()
    main(args.params)