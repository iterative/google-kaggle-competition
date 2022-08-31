import argparse
from operator import index
from zipfile import ZipFile
from pathlib import Path
import yaml


def main(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    data_root = Path(params["data"]["root"])

    dataset_path = data_root / cli_params.dataset
    zipfile = ZipFile(dataset_path.with_suffix(".zip"))
    files = zipfile.namelist() 
    index_file_name = f"{cli_params.dataset}_{params['data']['index_file']}"
    with open(data_root / index_file_name, "w") as f:
        f.write("\n".join(files))
    zipfile.extractall(path=dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)
