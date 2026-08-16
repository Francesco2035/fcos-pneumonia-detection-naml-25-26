from torch.utils.data import Dataset
import pydicom

from pathlib import Path


class DICOMDataset(Dataset):

    def __init__(
        self,
        dcm_path,
        transform=None,
    ):
        super().__init__()

        self.dcm_path = Path(dcm_path)
        self.transform = transform

        # Lista dei file DICOM presenti nella directory
        self.image_paths = []

        self._index_dicom_files()



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, index):

        path = self.image_paths[index]

        image = self._load_dicom(path)

        if self.transform is not None:
            image = self.transform(image)

        return image



    def _index_dicom_files(self):

        # Directory contenente i DICOM
        path = self.dcm_path

        # Cerca solamente i file .dcm
        self.image_paths = sorted(
            path.glob("*.dcm")
        )


    def _load_dicom(self, path):

        dicom = pydicom.dcmread(path)

        image = dicom.pixel_array

        return image




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    test_dcm_path = (
        "data/rsna-pneumonia-detection-challenge/"
        "stage_2_train_images"
    )

    dataset = DICOMDataset(
        dcm_path=test_dcm_path,
        transform=None,
    )

    print(
        "Number of DICOM files:",
        len(dataset),
    )

    image = dataset[1]

    print(
        "Shape:",
        image.shape,
    )

    print(
        "Dtype:",
        image.dtype,
    )

    print(
        "Min:",
        image.min(),
    )

    print(
        "Max:",
        image.max(),
    )

    plt.imshow(
        image,
        cmap="gray",
    )

    plt.axis("off")

    plt.show()

    print(dataset.__len__())