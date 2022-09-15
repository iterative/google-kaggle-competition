import fiftyone as fo
import fiftyone.utils.random as fo_random
from pathlib import Path


def export_dataset(dataset, path):
    dataset.export(export_dir=path, 
                dataset_type=fo.types.FiftyOneImageClassificationDataset,
                label_field='ground_truth',
                data_path='manifest.json',
                labels_path='labels.json',
                export_media='manifest',
                overwrite=True
                )


def main():
    data_root = Path("data")
    dataset_name = "kaggle_130k"
    dataset = fo.Dataset(dataset_name, overwrite=True)
    dataset.add_dir(data_root / dataset_name, dataset_type=fo.types.dataset_types.FiftyOneImageClassificationDataset)
    view_train,view_val,view_test = fo_random.random_split(dataset, [0.6,0.2,0.2], seed=51)

    export_dataset(view_train, str(data_root / dataset_name /'train'))
    export_dataset(view_val, str(data_root / dataset_name / "val"))
    export_dataset(view_test, str(data_root / dataset_name / "test"))
    
if __name__=="__main__":
    main()