#!/bin/sh

dvc get https://github.com/iterative/google-kaggle-competition-data-pipeline datasets/kaggle_130k/labels.json --rev a3ba6fb
dvc get https://github.com/iterative/google-kaggle-competition-data-pipeline datasets/kaggle_130k/kaggle_130k.zip --rev a3ba6fb

unzip kaggle_130k.zip