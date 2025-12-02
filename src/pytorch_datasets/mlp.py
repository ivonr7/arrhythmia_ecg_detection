import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset 
from .annot_dataset import Annot_Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

#to control the random variables so graphs stay the same o
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ECG_MLP(nn.Module):

    
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev,num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def load_dataset(split="train", batch_s=16, limit=None):
    CSV_ROOT = "/Users/madi/arrhythmia_ecg_detection/dataset"
    RAW_ROOT = "/Users/madi/dataset"
    #annotations folder
    extracted_folder = f"{CSV_ROOT}/{split}" 
    #metadata + annotations
    dataset_folder = CSV_ROOT

    ds = Annot_Dataset(
        extracted_folder= extracted_folder,
        dataset_folder= dataset_folder,
        window_duration=0.3,
        classification_task="binary",
        train=(split == "train"),
    )

    ds.signal_folder = Path(RAW_ROOT)

    if limit is not None:
        ds = Subset(ds, range(limit))

    #take the mean over window chanels
    def collate_fn(batch):
        Xs, Ys = zip(*batch)

        Xs = [x.mean(dim=0).float() for x in Xs]
        Xs = torch.stack(Xs)

        Ys = torch.tensor(Ys, dtype=torch.long)
        return Xs, Ys
    
    return DataLoader(
        ds,
        batch_size = batch_s,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )


def confusion_counts(preds, labels):
    """
    preds  : predicted class indices (0 or 1)
    labels : true class indices (0 or 1)

    """
    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()
    return TP, FP, FN, TN

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
   # total_TP = total_FP = total_FN = total_TN = 0

    #loop = tqdm(dataloader, desc="Training")
    #for X, y in loop:
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

        #TP, FP, FN, TN = confusion_counts(preds, y)
        #total_TP += TP
        #total_FP += FP
       # total_FN += FN
        #total_TN += TN

    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100
    return avg_loss, accuracy

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_TP = total_FP = total_FN = total_TN = 0
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            TP, FP, FN, TN = confusion_counts(preds, y)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100

    return avg_loss, accuracy, total_TP, total_FP, total_FN, total_TN


def plot_graph(train_hist, test_hist, epochs):
    fix, ax = plt.subplots(1, 2, figsize=(14, 5))

    #Accuracy
    ax[0].plot(range(1, epochs + 1), train_hist["acc"], label="Train Accuracy")
    ax[0].plot(range(1, epochs + 1), test_hist["acc"], label="Test Accuracy")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    #Loss
    ax[1].plot(range(1, epochs + 1), train_hist["loss"], label="Train Loss")
    ax[1].plot(range(1, epochs + 1), test_hist["loss"], label="Test Loss")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cpu")
    train_loader = load_dataset(split="train", batch_s=16)
    test_loader = load_dataset(split="test", batch_s=16)
    #added
    print("Train samples:", len(train_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    xb, yb = next(iter(train_loader))
    input_size = xb.shape[1]
    print("input size = ", input_size)

    model = ECG_MLP(
        input_size = input_size,
        hidden_sizes = [128,64],
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    EPOCHS = 3
    train_hist = {"loss": [], "acc": []}
    test_hist = {"loss": [], "acc": []}

    print("\ntraining...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        #tr_loss, tr_acc, tr_TP, tr_FP, tr_FN, tr_TN = train_epoch(
        #model, train_loader, criterion, optimizer, device)
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        te_loss, te_acc, te_TP, te_FP, te_FN, te_TN = eval_epoch(
            model, test_loader, criterion, device
        )
        
        #te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")

        print(f"Test:  loss={te_loss:.4f}, acc={te_acc:.4f}")
        print(f"TP: {te_TP} | FP: {te_FP} | FN: {te_FN} | TN: {te_TN}")

        #total_cm = tr_TP + tr_FP + tr_FN + tr_TN
        #print("Train Confusion Matrix Total =", total_cm)

        total_cm_test = te_TP + te_FP + te_FN + te_TN
        print("Test Confusion Matrix Total =", total_cm_test)

        train_hist["loss"].append(tr_loss)
        train_hist["acc"].append(tr_acc)
        test_hist["loss"].append(te_loss)
        test_hist["acc"].append(te_acc)
    torch.save(model.state_dict(), "mlp_baseline.pt")

    plot_graph(train_hist, test_hist, EPOCHS)

    # Use the confusion matrix from the LAST epoch
    TP = te_TP
    FP = te_FP
    FN = te_FN
    TN = te_TN

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("\nFinal Confusion Matrix:")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")

    print("\nFinal Evaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
