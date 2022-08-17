### How to install and run

1. `poetry install`
2a. `poetry run dvc pull`
3a. `poetry run dvc repro`

or alternatively:

2b. `poetry shell` which activates the environment
3b. `dvc pull`
4b. `dvc repro` 




### DVC pipeline
We are using DVC pipeline to run and manage experiments. The definition of the pipeline consists of two parts:
* `dvc.yaml` --- main file defining stages with: cli command, dependencies, outputs, metrics and plots
* `params.yaml` --- parameters of the pipeline such as paths/model hyperparams

Each stage is defined in a separate python file in `src/stages`. All helper functions and classes are in `src/utils`.
When we run `poetry install` it installs this repo as python package so we can use it without adding it explicitly to PYTHONPATH.


### Dataset

For the benchmark purposes we are using [this](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/336574) starter dataset.
We downloaded this dataset and added it to DVC remote in our S3 bucket. The command `dvc pull` downloads the dataset to `data/teacher_data/Benchmark.zip`.
There are two stages of the pipeline: unzip_dataset and split_dataset that unzip a zipped dataset and split it into `train` and `val`. 

#### How to add a new dataset
The code expects the following structure of a zip file:
```
Dataset.zip
|-Dataset
|---class1/
|------img1
|------img2
|---class2/
|------img1
|------img2
....
```

This zip file should be placed to `data/teacher_data/` then add `Dataset`(or any name of <dataset>.zip) to `teacher_models` list in `params.yaml` and that's it.  


### Metrics and visualisations
All metrics and visualisisations can be found under `reports/`. 
Folders that have prefix `dvclive` are produced by DVClive plugin that tracks live metrics of model training. `reports/plots` directory contains embeddings for all datasets listed in params.yaml under `teacher_models`



### Notes

* We used this implementation: [link](https://github.com/adambielski/siamese-triplet) 
* We had to modify dataloaders to use images stored in folders
* model has been changed from small custom CNN to efficientnet_b3 with pretrained weights
* more experiments are needed to see if it's benefitial. 
 

