from torch.utils.data import Dataset
import torch
import wfdb as wb
from pathlib import Path
import pandas as pd
import numpy as np

class Annot_Dataset(Dataset):
    def __init__(
            self,extracted_folder:str,
            dataset_folder:str,
            window_duration:float = 0.3,
            classification_task = 'binary'
        ):
        super().__init__()
        self.meta,self.annot = Annot_Dataset.get_dataset(extracted_folder)
        self.labels = {
            label:i for i,label in enumerate(self.annot['labels'].unique())
        }
        self.nclasses = len(list(self.labels.keys()))
        self.signal_folder = Path(dataset_folder)
        self.window_duration = window_duration
        self.task = classification_task
    

    def __getitem__(self, index:int, n_channels:int = 12):
        """
            Get Single Training Example
            From data exploration there are NA values
            for some patients on sensors 12-19
            Therefore we won't use them for training
            Returns:
            X: Signal Voltage Matrix of Shape (window_size,n_channels)
            Y: tensor of size (n_classes) with the correct class 
            set to 1
        """
        annot_row = self.annot.loc[index]
        fs = annot_row['frequency']
        window_size = int(fs * self.window_duration)
        window = Annot_Dataset.compute_window(
            self.signal_folder / annot_row['file_name'],
            index = annot_row['indicies'],
            window_size=window_size
        )
        labels = torch.zeros(size = (self.nclasses,))
        idx = self.labels[annot_row['labels']]
        labels[idx] = 1
        return torch.tensor(window[:,:n_channels]), labels
    
    def __len__(self):
        return self.annot.shape[0]
    @staticmethod
    def get_dataset(dataset_folder:str):
        """
            This function reads the output
            of the scripts/combine_dataset.py script
            returns a tuple containing (metadata,annotations) 
            dataframes
        """
        meta = pd.read_csv(
            Path(dataset_folder) / "patient_metadata.csv"
        )
        annot = pd.read_csv(
            Path(dataset_folder) / "patient_annotations.csv"
        )
        return meta, annot
    @staticmethod    
    def compute_window(record:str,index:int,window_size:int = 294):
        """
            This Function Extracts the signal window around a an annotation 
            building off yasnas code. If the signal window is OOB pad with zero to 
            retain window size
            returns data window as a numpy array
        """
        signal_min = index - window_size
        signal_max = index + window_size
        padding = 0
        if signal_min < 0:
            padding= abs(signal_min)
            signal_min = 0
        record = wb.rdrecord(record,sampfrom=signal_min,sampto=signal_max)
        return np.pad(record.p_signal,((padding,0), (0,0)))
    

if __name__ == "__main__":
    ds = Annot_Dataset(
        extracted_folder="dataset",
        dataset_folder="physionet.org/files/leipzig-heart-center-ecg/1.0.0",
        window_duration=0.3
    )
    print(ds[1000][0].shape)
