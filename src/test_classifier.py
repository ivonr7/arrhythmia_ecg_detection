import torch
from torch.utils.data import dataloader
from models.arrformer import ArrFormer
from pytorch_datasets.annot_dataset import Annot_Dataset
from argparse import ArgumentParser
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd

def parse_args():
    parser = ArgumentParser("Test Classification Performance of Model")
    parser.add_argument("--window-size", type=float,default=0.3, help='Window Size of Signal')
    parser.add_argument('model',type=str, help="Model .pt filt to use")
    parser.add_argument("out", type=str, help="Where to write scores")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ds = Annot_Dataset(
        extracted_folder='dataset',
        dataset_folder='physionet.org/files/leipzig-heart-center-ecg/1.0.0',
        window_duration=args.window_size,
        train=False
    )
    # Load Model and Set to Test Mode
    state = torch.load(args.model)
    model = ArrFormer(dim = 12)
    model.load_state_dict(state)
    model.to('cuda')
    model.eval()
    c = Counter(["TP","FP", "TN", "FN"])
    for i in tqdm(range(len(ds)),desc = "Predicting Test Data"):
        X,y = ds[i]
        X = X.float().unsqueeze(0).to("cuda")
        y_hat = model(X)
        pred = y_hat.argmax().item()

        # For Generating Confusion Matrix
        if y == 1:
            match pred:
                case 1:
                    c['TP'] += 1
                case 0: 
                    c["FP"] += 1
        else:
            match pred:
                case 1:
                    c['FN'] += 1
                case 0: 
                    c["TN"] += 1
    
    scores = pd.Series(dict(c))
    scores.to_json(args.out)

