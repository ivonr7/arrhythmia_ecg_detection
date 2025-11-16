from torch.utils.data import Dataset
import torch
import wfdb as wb
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

NORMAL_DICT = {
    'N',
    'R',
    'L',
    'bBI',
    'j',
    '//A',
    '//V'
    'f'
}
BEAT_DICT = {
    'N',
    'R',
    'L',
    'b',
    'j',
    'X',
    'A',
    'a',
    'V',
    'F',
    'J',
    '/',
    'f'
}


@dataclass
class Window:
    midpoint:int
    start:int
    end:int
    padding:tuple

class Annot_Dataset(Dataset):
    def __init__(
            self,extracted_folder:str,
            dataset_folder:str,
            window_duration:float = 0.3,
            classification_task = 'binary'
        ):
        super().__init__()
        self.meta,self.annot = Annot_Dataset.get_dataset(extracted_folder)
        beat_mask = np.array([
            True if l in BEAT_DICT else False \
                for l in self.annot['labels']
        ])        
        self.beats = self.annot.loc[beat_mask,:].reset_index(drop=True).copy() 
        self.beats['combined_label'] = self.beats['labels'] + self.beats['aux_note'].fillna("")
        
        if classification_task != 'binary':
            self.labels2vec = {
                label:i for i,label in enumerate(self.beats['combined_label'].unique())
            }
            self.vec2labels = {
                i:label for i,label in enumerate(self.beats['combined_label'].unique())
            }
        else:
            self.t = 0.5
        self.nclasses = len(list(self.beats['combined_label'].unique()))
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
        annot_row = self.beats.loc[index,:]
        fs = annot_row['frequency']
        window_size = int(fs * self.window_duration)
        window = Annot_Dataset.compute_window(
            annot_row['length'],
            index = annot_row['indicies'],
            window_size=window_size
        )
        record = wb.rdrecord(
            self.signal_folder / annot_row['file_name'],
            sampfrom=window.start,
            sampto=window.end
        )
        X = np.pad(record.p_signal,(window.padding, (0,0)))
        
        if self.task == 'binary':
            labels = 0 if annot_row['combined_label'] in NORMAL_DICT else 1
        else:
            labels = torch.zeros(size = (self.nclasses,))
            idx = self.labels2vec[annot_row['combined_label']]
            labels[idx] = 1
        return torch.tensor(X[:,:n_channels]), labels
    
    def __len__(self):
        return self.labels.shape[0]
    def str_class(self,y):
        if self.task == 'binary':
            return self.y > self.t
        index = torch.argmax(y)
        return self.vec2labels[index]
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
    def compute_window(length:int,index:int,window_size:int = 294):
        """
            This Function Extracts the signal window around a an annotation 
            building off yasnas code. If the signal window is OOB pad with zero to 
            retain window size
            returns data window as a window object 
        """
        signal_min = index - window_size
        signal_max = index + window_size
        padding = [0,0]
        if signal_min < 0:
            padding[0] = abs(signal_min)
            signal_min = 0
        if signal_max > length:
            padding[1] = signal_max - length
            signal_max = signal_max - padding[1]
        return Window(index,signal_min,signal_max,padding)
    

if __name__ == "__main__":
    ds = Annot_Dataset(
        extracted_folder="dataset",
        dataset_folder="physionet.org/files/leipzig-heart-center-ecg/1.0.0",
        window_duration=0.3,
        classification_task='binary'
    )
    for X,y in ds:
        print(X,y)