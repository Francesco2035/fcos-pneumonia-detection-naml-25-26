from src.datasets.DICOMDataset import DICOMDataset
import pandas as pd


class RSNAPneumoniaDataset(DICOMDataset):

    def __init__(
        self,
        dcm_path,
        csv_path,
        transform=None,
    ):

        super().__init__(
            dcm_path=dcm_path,
            transform=transform,
        )

        self.csv_path = csv_path
        self.df = None
        self.annotations = None

        self._read_csv()


    def _read_csv(self):
        self.df = pd.read_csv(self.csv_path)



csv_path = "data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"


train_dcm_path = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)

dataset = RSNAPneumoniaDataset(train_dcm_path,csv_path, None)

print(dataset.df)

