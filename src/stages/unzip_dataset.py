import argparse
from zipfile import ZipFile
from pathlib import Path
import yaml


def main(params):
    stage_params = yaml.safe_load(open(params.params))
    data_root = Path(stage_params["data"]["root"])

    output_path = data_root
    if params.teacher:
        teacher_data_path = stage_params["data"]["teacher_data"]
        output_path /= teacher_data_path
        dataset_path = (
            data_root / teacher_data_path / params.teacher)
    else:
        dataset_path = data_root / stage_params["data"]["train"]
        
    zipfile = ZipFile(dataset_path.with_suffix(".zip"))
    zipfile.extractall(path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--teacher", type=str)
    args = parser.parse_args()
    main(args)
