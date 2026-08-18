from src.datasets.DICOMDataset import DICOMDataset
import pandas as pd
from torchvision import tv_tensors
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader



def detection_collate_fn(batch):

    images, targets = zip(*batch)

    images = torch.stack(images)

    targets = list(targets)

    return images, targets







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
        self.annotations = {}

        self._read_csv()
        self._build_annotations()


    def _read_csv(self):
        self.df = pd.read_csv(self.csv_path)


    
    def _build_annotations(self):
        groupById = self.df.groupby("patientId")
        for patient_id, data in groupById:

            labels = []
            boxes = []

            for _, row in data.iterrows():

                if row["Target"] == 1:

                    labels.append(1)

                    boxes.append([
                        row["x"],
                        row["y"],
                        row["x"] + row["width"],
                        row["y"] + row["height"],
                    ])

            self.annotations[patient_id] = {
                "labels": labels,
                "boxes": boxes,
            }


    def __getitem__(self, index):

            # -----------------------------
            # DICOM
            # -----------------------------

            path = self.image_paths[index]

            patient_id = path.stem

            image = self._load_dicom(path)

            # -----------------------------
            # Image -> tv_tensor Image
            # -----------------------------

            image = v2.ToImage()(image)

            # -----------------------------
            # Annotation
            # -----------------------------

            annotation = self.annotations[patient_id]

            boxes = torch.tensor(
                annotation["boxes"],
                dtype=torch.float32,
            ).reshape(-1, 4)

            labels = torch.tensor(
                annotation["labels"],
                dtype=torch.int64,
            )

            # -----------------------------
            # Bounding boxes
            # -----------------------------

            boxes = tv_tensors.BoundingBoxes(
                boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=image.shape[-2:],
            )

            # -----------------------------
            # Target
            # -----------------------------

            target = {
                "boxes": boxes,
                "labels": labels,
            }

            # -----------------------------
            # Transform
            # -----------------------------

            if self.transform is not None:

                image, target = self.transform(
                    image,
                    target,
                )

            return image, target
    

    def get_dataloader(
        self,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=detection_collate_fn,
        )
    
