from scripts.ecg_dataset import ECGDataset

dataset = ECGDataset(
    ecg_path="data/patients_ecg.csv",
    ann_path="data/patient_annotations.csv",
    meta_path="data/patient_metadata.csv",
    label_mode="binary",
    window_sec=5.0
)

print("Dataset length:", len(dataset))

x, y = dataset[500]
print("Sample shape:", x.shape)
print("Label:", y)
