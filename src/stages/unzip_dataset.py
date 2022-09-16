import argparse
from operator import index
from zipfile import ZipFile
from pathlib import Path
import yaml
import shutil


def main(cli_params):
    params = yaml.safe_load(open(cli_params.params))
    data_root = Path(params["data"]["root"])

    dataset_path = (data_root / cli_params.dataset / cli_params.dataset).with_suffix(
        ".zip"
    )
    zipfile = ZipFile(dataset_path)
    files = zipfile.namelist()
    index_file_name = f"{cli_params.dataset}_{params['data']['index_file']}"
    with open(data_root / index_file_name, "w") as f:
        f.write("\n".join(files))
    labels_file = data_root / cli_params.dataset / "labels.json"

    zipfile.extractall(path=data_root / params["data"]["training"] / cli_params.dataset)
    shutil.copy(
        labels_file,
        data_root / params["data"]["training"] / cli_params.dataset / "labels.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)
