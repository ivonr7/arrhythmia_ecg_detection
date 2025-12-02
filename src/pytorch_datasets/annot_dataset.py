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
    midpoint: int
    start: int
    end: int
    padding: tuple

class Annot_Dataset(Dataset):
    def __init__(
        self,
        extracted_folder: str,
        dataset_folder: str,
        window_duration: float = 0.3,
        classification_task='binary',
        train=True
    ):
        super().__init__()
        self.meta, self.annot = Annot_Dataset.get_dataset(dataset_folder, train)

        beat_mask = np.array([
            (l in BEAT_DICT) for l in self.annot["labels"]
        ])
        self.beats = self.annot.loc[beat_mask, :].reset_index(drop=True).copy()
        self.beats["combined_label"] = (
            self.beats["labels"] + self.beats["aux_note"].fillna("")
        )

        if classification_task != 'binary':
            unique_labels = list(self.beats['combined_label'].unique())
            self.labels2vec = {label: i for i, label in enumerate(unique_labels)}
            self.vec2labels = {i: label for i, label in enumerate(unique_labels)}
        else:
            self.t = 0.5

        self.nclasses = len(list(self.beats["combined_label"].unique()))
        self.signal_folder = Path("/Users/madi/dataset")


        self.window_duration = window_duration
        self.task = classification_task

    def __getitem__(self, index: int, n_channels: int = 12):

        annot_row = self.beats.loc[index, :]
        fs = annot_row["frequency"]
        window_size = int(fs * self.window_duration)

        window = Annot_Dataset.compute_window(
            annot_row["length"],
            index=annot_row["indicies"],
            window_size=window_size
        )

        # read ECG window
        record = wb.rdrecord(
            self.signal_folder / annot_row["file_name"],
            sampfrom=window.start,
            sampto=window.end
        )

        X = np.pad(record.p_signal, (window.padding, (0, 0)))

        if self.task == 'binary':
            label = 0 if annot_row["combined_label"] in NORMAL_DICT else 1
            return torch.tensor(X[:, :n_channels]), label


        y = torch.zeros(size=(self.nclasses,))
        idx = self.labels2vec[annot_row["combined_label"]]
        y[idx] = 1
        return torch.tensor(X[:, :n_channels]), y

    def __len__(self):
        return len(self.beats)

    @staticmethod
    @staticmethod
    @staticmethod
    def get_dataset(dataset_folder: str, istrain: bool = True):
        """
        Load metadata + annotation CSVs

        Expected structure:
            dataset/train/patient_annotations.csv
            dataset/train/patient_metadata.csv
            dataset/test/patient_annotations.csv
            dataset/test/patient_metadata.csv
        """
        split = "train" if istrain else "test"

        meta_path  = Path(dataset_folder) / split / "patient_metadata.csv"
        annot_path = Path(dataset_folder) / split / "patient_annotations.csv"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")

        if not annot_path.exists():
            raise FileNotFoundError(f"Annotations file missing: {annot_path}")

        meta  = pd.read_csv(meta_path)
        annot = pd.read_csv(annot_path)

        return meta, annot


    @staticmethod
    def compute_window(length: int, index: int, window_size: int = 294):
        signal_min = index - window_size
        signal_max = index + window_size
        padding = [0, 0]

        if signal_min < 0:
            padding[0] = -signal_min
            signal_min = 0

        if signal_max > length:
            padding[1] = signal_max - length
            signal_max = length

        return Window(index, signal_min, signal_max, padding)


if __name__ == "__main__":
    ds = Annot_Dataset(
        extracted_folder="dataset",
        dataset_folder="dataset",
        window_duration=0.3,
        classification_task="binary"
    )

    print("Dataset length =", len(ds))
    X, y = ds[0]
    print("Sample X shape:", X.shape)
    print("Label:", y)
