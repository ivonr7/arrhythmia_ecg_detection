import sys
from pathlib import Path

# Fix import paths - adjust if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
# Will need to adjust if everything is contained in the project folder
# Currently, the model and the combined datasets are in separate folders, outside the arrhythmia project folder
sys.path.append(str(PROJECT_ROOT / "arrhythmia_ecg_detection" / "src" / "pytorch_datasets"))

from annot_dataset import Annot_Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random, numpy as np

from RNN import ECG_RNN #RNN model in RNN.py

# For debugging and controlling randomness - can remove if needed
SEED = 22
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)


# Paths - adjust if needed
DATA_PATH = PROJECT_ROOT / "combined_data"

wfdb_path = PROJECT_ROOT / "arrhythmia_ecg_detection" / "physionet.org" / "files" / "leipzig-heart-center-ecg" / "1.0.0"


# Hyperparameters
INPUT_SIZE = 12
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 64 # tried 32
NUM_EPOCHS = 4 # tried 6, 10, 15
LR = 0.001


# Load datasets
train_ds = Annot_Dataset(
    extracted_folder=DATA_PATH,
    dataset_folder=wfdb_path,
    window_duration=0.3,
    classification_task="binary",
    train=True
)

test_ds = Annot_Dataset(
    extracted_folder=DATA_PATH,
    dataset_folder=wfdb_path,
    window_duration=0.3,
    classification_task="binary",
    train=False
)


print("Train samples:", len(train_ds.beats))
print("Test samples:", len(test_ds))

NUM_CLASSES = 1
print("Number of classes:", NUM_CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# Model
model = ECG_RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(NUM_EPOCHS):

    # TRAIN
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X = X.float().to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()
        targets = y.int()
        correct += (preds == targets).sum().item()
        total += targets.numel()

    train_loss /= len(train_loader)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # TEST
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.float().to(device)
            y = y.to(device).float().unsqueeze(1)

            outputs = model(X)
            loss = criterion(outputs, y)

            test_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            targets = y.int()

            correct += (preds == targets).sum().item()
            total += targets.numel()

            TP += ((preds == 1) & (targets == 1)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total

    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}\n")
    print(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")



# Save model
torch.save(model.state_dict(), "ecg_rnn_model.pt")
print("Model saved as ecg_rnn_model.pt")

# PLOTS
# Plot results
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, test_accs, label="Test Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
