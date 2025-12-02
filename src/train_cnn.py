import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_datasets.annot_dataset import Annot_Dataset
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



class ArrCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64 * 586, num_classes)
    def forward(self,x):
        x= x.permute(0,2,1)
        x=self.relu1(self.conv1(x))
        x=self.relu2(self.conv2(x))
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
def datasetloader():
    #ds = Annot_Dataset(
        #extracted_folder="data/combined",
        #dataset_folder="data",
        #window_duration=0.3,
        #classification_task="binary"
    #)

    train_ds = Annot_Dataset(
        extracted_folder="data/train",
        dataset_folder="data",
        window_duration=0.3,
        classification_task="binary"
    )
    test_ds = Annot_Dataset(
        extracted_folder="data/test",
        dataset_folder="data",
        window_duration=0.3,
        classification_task="binary"
    )
    x0, y0 = train_ds[0]
    return train_ds, test_ds

def train_model(model,loader,criterion,optimizer,device):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for X,y in tqdm(loader, desc="Train", leave=False):
        X=X.float().to(device)
        y=y.long().to(device)
        optimizer.zero_grad()
        logits=model(X)
        loss=criterion(logits,y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        pred=torch.argmax(logits,dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
    avg_loss = running_loss/len(loader)
    acc=correct/total
    return avg_loss, acc
def evaluate(model,loader,criterion,device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for X,y in tqdm(loader, desc="Test", leave=False):
            X=X.float().to(device)
            y=y.long().to(device)
            logits=model(X)
            loss=criterion(logits,y)
            running_loss+=loss.item()
            pred=torch.argmax(logits,dim=1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    avg_loss = running_loss/len(loader)
    acc=correct/total
    return avg_loss, acc

def confusion_counts(model, loader, device):
    model.eval()
    TP=FP=TN=FN=0
    with torch.no_grad():
        for X, y in loader:
            X=X.float().to(device)
            y=y.long().to(device)

            logits=model(X)
            preds=torch.argmax(logits,dim=1)
            for pred, true in zip(preds,y):
                if true==1:
                    if pred==1:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if pred==0:
                        TN+=1
                    else:
                        FP+=1
    return TP, FP,TN,FN
if __name__ == "__main__":
    #datasetfull = datasetloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = datasetloader()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    #num_classes= test_ds.nclasses
    model = ArrCNN(num_classes=2).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs=10
    train_losses=[]
    train_accs=[]
    test_losses=[]
    test_accs=[]

    best_test_loss = float('inf')
    patience=3
    patience_counter =0
    best_state_dict = None

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(
            f"Train Loss:{train_loss:.4f}|Train Accuracy: {train_acc:.4f}|"
            f"test Loss:{test_loss:.4f}|test Accuracy: {test_acc:.4f}"
        )
        if test_loss < best_test_loss:
            print(f"improved")
            best_test_loss = test_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter+=1
            print(f"No improvement")
            if patience_counter>=patience:
                print(f"Early stopping")
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(),"arrhythmia_cnn_bianry.pt")
        
    best_model=ArrCNN(num_classes=2).to(device)
    best_model.load_state_dict(torch.load("arrhythmia_cnn_bianry.pt"))
    best_model.eval()
    TP, FP, TN, FN = confusion_counts(best_model, test_loader,device)
    print("Confusion Matrix:")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1=2*precision*recall/(precision+recall)
    accuracy= (TP+TN)/(TP+TN+FP+FN)
    print("\nMetrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    epochs = range(1,len(train_losses)+1)
    plt.subplot(1,2,1)
    plt.plot(epochs, train_accs,label="Train Accuracy")
    plt.plot(epochs, test_accs,label="Test Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Accuracy")
    plt.ylabel("Epoch")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_losses,label="Train Loss")
    plt.plot(epochs, test_losses,label="Test Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Loss")
    plt.ylabel("Epoch")
    plt.legend()
    plt.show()


