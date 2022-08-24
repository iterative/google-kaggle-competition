import argparse
from zipfile import ZipFile
from pathlib import Path
import yaml


def main(params):
    stage_params = yaml.safe_load(open(params.params))
    data_root = Path(stage_params["data"]["root"])
    teacher_data_path = stage_params["data"]["teacher_data"]

    if params.teacher:
        teacher_data_archive_path = (
            data_root / teacher_data_path / params.teacher
        ).with_suffix(".zip")
        zipfile = ZipFile(teacher_data_archive_path)
        zipfile.extractall(path=data_root / teacher_data_path)
    else:
        baseline_dataset = stage_params["data"]["train"]
        dataset_dir = data_root / baseline_dataset
        dataset_dir.mkdir(exist_ok=True)
        zipfile = ZipFile(dataset_dir.with_suffix(".zip"))
        zipfile.extractall(path=dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--teacher", type=str)
    args = parser.parse_args()
    main(args)
