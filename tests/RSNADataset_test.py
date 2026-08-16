from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset


csv_path = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

train_dcm_path = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)


dataset = RSNAPneumoniaDataset(
    dcm_path=train_dcm_path,
    csv_path=csv_path,
    transform=None,
)

print(dataset.df.head())
print(dataset.df.columns)
print(dataset.df.shape)