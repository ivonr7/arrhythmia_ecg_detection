import polars as pl
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):

    def __init__(
        self,
        ecg_path: str,
        ann_path: str,
        meta_path: str,
        window_sec: float = 5.0,
        sample_rate: float = 977.0,
        label_mode: str = "binary",
        leads=None,
    ):
        self.ecg_path = ecg_path
        self.ann_path = ann_path
        self.meta_path = meta_path
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.label_mode = label_mode
        self.leads = leads

        # Loading files using polars
        self.ecg_lazy = pl.scan_csv(ecg_path)
        self.ann_lazy = pl.scan_csv(ann_path)
        self.meta = pl.read_csv(meta_path)

        # Collect annotation info into memory 
        self.ann = self.ann_lazy.collect()
        self.samples = self.ann["sample"].to_list()
        self.labels = self.ann["symbol"].to_list()
        #print("First 20 symbols in annotations:")
        #print(self.ann["symbol"].head(20))

        # trying to store subject IDs for grouping --> to be improved
        if "file_name" in self.ann.columns:
            self.subjects = self.ann["file_name"].to_list()
        else:
            self.subjects = ["unknown"] * len(self.samples)

        print(f"Loaded {len(self.samples)} annotated beats.")

    def __len__(self):
        return len(self.samples) 

    def _get_window(self,file_name, start_sample: int):
        
        #Choose a ECG window for a given patient
        half_window = int((self.window_sec * self.sample_rate) / 2)
        start = max(0, start_sample - half_window)
        end = start_sample + half_window

        # Filter for this patient's data only
        df = (
            self.ecg_lazy
            .slice(offset=start, length=end - start)
            .collect()
        )

        # take all leads(signal chanels)
        leads = [c for c in df.columns if c != "time_s"]
        signal = (
            df.select(leads)
            .fill_null(0)  # replace missing values with 0
            .with_columns([pl.all().cast(pl.Float32)])  # ensure all columns are numeric
            .to_numpy()
            .T
        )

        return signal

    def __getitem__(self, idx):
        file_name = self.subjects[idx]
        sample = self.samples[idx]
        label_text = self.labels[idx]
        '''
        # Binary label mapping
        if self.label_mode == "binary":
            if label_text is None:
                label = 0  # Default to normal if missing
            else:
                label = 0 if label_text == "N" else 1
        else:
            label = label_text if label_text is not None else "Unknown"
        '''
        signal = self._get_window(file_name, sample)
        x = torch.tensor(signal, dtype=torch.float32)
        #y = torch.tensor(label, dtype=torch.long)
        y = label_text

        return x, y
