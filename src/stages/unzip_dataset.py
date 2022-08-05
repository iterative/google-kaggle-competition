import argparse
from zipfile import ZipFile
from pathlib import Path
import yaml


def main(params):
    stage_params = yaml.safe_load(open(params))
    data_root = Path(stage_params["data"]['data_root'])
    dataset_path = stage_params["data"]["dataset_baseline"]

    zipfile = ZipFile(data_root / stage_params["data"]["archive_baseline"] )
    zipfile.extractall(path=data_root)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='params.yaml')
    args = parser.parse_args()
    main(args.params)