### Disclaimer

* We used this implementation: [link](https://github.com/adambielski/siamese-triplet) 
* We modified dataloaders to use images stored in folders
* model has been changed from small custom CNN to efficientnet_b3 with pretrained weights


### How to install and run

1. `poetry install`
2. `poetry shell` which activates the environment
3. `dvc import https://github.com/iterative/google-kaggle-competition-data-pipeline data/baseline_split.zip -o data/` -- it will import baseline dataset from our data registry. 
4. `dvc repro` 

If dataset is imported but the data were removed/corrupted, it's enough to run `dvc pull data/baseline_split.zip.dvc` to pull data from the remote.



### DVC pipeline
We are using DVC pipeline to run and manage experiments. The definition of the pipeline consists of two parts:
* `dvc.yaml` --- main file defining stages with: cli command, dependencies, outputs, metrics and plots
* `params.yaml` --- parameters of the pipeline such as paths/model hyperparams

Each stage is defined in a separate python file in `src/stages`. All helper functions and classes are in `src/utils`.
When we run `poetry install` it installs this repo as python package so we can use it without adding it explicitly to PYTHONPATH.


### Dataset

For the baseline purposes we are using [this](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/336574) starter dataset.
We downloaded this dataset and added it to DVC remote in our S3 bucket. The command `dvc pull data/baseline_split.zip.dvc` downloads the dataset to `data/baseline_split.zip`.

#### How to add a new dataset
If you would like to run the pipeline with a custom dataset please refer to this [repo](https://github.com/iterative/google-kaggle-competition-data-pipeline/). It has scripts to prepare data into format that the training pipeline expects. 


The produced zip file should be placed to `data/` then update `params.yaml` section `train` with archive name without the extension `.zip`


### Metrics and visualisations
All metrics and visualisisations can be found under `reports/`. 
 `dvclive` is produced by DVClive plugin that tracks live metrics of model training. `reports/plots` directory contains visualisations of embeddings.




 

