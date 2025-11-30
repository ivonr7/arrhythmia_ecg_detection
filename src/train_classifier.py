from pathlib import Path
import torch
from torch.optim import Adam
import torchvision.transforms as tv
from torchaudio import transforms as ta
import wandb as wb
from models.arrformer import ArrFormer
from pytorch_datasets.annot_dataset import Annot_Dataset,get_validation_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from argparse import ArgumentParser
from accelerate import Accelerator
'''
    Perform Training Epoch
'''
def single_epoch(model,dl, optim, loss_func,device, idx = 0):
    model.to(device)

    running_loss = 0
    data = tqdm(dl, desc=f"Epoch:{idx}")

    for batch in data: 
        optim.zero_grad()

        X,y  = batch
        X = X.to(device).float()
        y = y.to(device).float()
        x_hat = model(X)
        loss = loss_func(x_hat, y.long())
        loss.backward()
        #device.backward()
        optim.step()
        running_loss += loss.item() / X.shape[0]
        data.set_postfix({"Loss" : loss})
    return running_loss

class TrainingTransforms:
    def __init__(self, transforms = None):
        self.transform = transforms
    def __call__(self, x):

        x = x.permute( 1, 0)
        
        if self.transform:
            x = self.transform(x)
        x=x.permute(1,0)
        return x

def get_args():
    parser =  ArgumentParser()
    parser.add_argument('--lr',type=float, default= 0.01, help='Learning Rate')
    parser.add_argument('--window-size',type=float, default=0.3, help='window size (s) of signal to consider')
    parser.add_argument('--epochs',type = int, default=10, help='number of epochs to train for')
    parser.add_argument('data',type=str,help='Location of .dat files')
    parser.add_argument('extracted',type=str,help='path to extracted annotation data')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with wb.init(
        entity='ivonr7-simon-fraser-university',
        project='arrythmia_classification',
        config={
            'window_size': args.window_size,
            'lr': args.lr,
            'stage': 'encoder',
            'epochs': args.epochs
        }
    ) as run:
        transforms = tv.RandomApply([
            ta.TimeMasking(40),
            ta.Vol(gain = 1.2, gain_type = "amplitude")
            ],p=0.4)

        # Define Classes
        transforms = TrainingTransforms(transforms)
        encoder = ArrFormer(dim = 12, n_heads = 4)
        ds = Annot_Dataset(
            extracted_folder='dataset',
            dataset_folder='physionet.org/files/leipzig-heart-center-ecg/1.0.0',
            window_duration=args.window_size,
            transforms = transforms
        )
        train_dl, val_dl = get_validation_split(
            ds,
            val_prop = 0.2
        )


        
        # Where to save trained model weights
        output = Path(wb.run.dir)
        output.mkdir(exist_ok = True, parents = True)
        
        # Load Data
        train_dl = DataLoader(train_dl,batch_size=32)
        val_dl = DataLoader(val_dl, batch_size=32)
        optim = Adam(encoder.parameters(),lr = args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        #accel = Accelerator()
        #model, optim, train_dl = accel.prepare(encoder,optim,train_dl)
        
        wb.watch(encoder)
        min_loss = 2*20**10
        max_correct = 0
        correct = 0
        for i in range(args.epochs):

            # Train For 1 Epoch
            encoder.train(True)
            loss = single_epoch(encoder, train_dl, optim,loss_fn, 'cuda' , i)
            run.log({'epoch':i,'avg_loss':loss})
            

            encoder.eval()
            v_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(val_dl, desc=f"Validation {i}"):
                X,y = batch
                X = X.to("cuda").float()
                y = y.to("cuda").long()
 
                x_hat = encoder(X)
                loss = loss_fn(x_hat,y)
                correct += (x_hat.argmax(axis =1) == y).sum()
                v_loss += loss.item() * X.size(0)
                total += y.size(0)

            run.log({'epoch':i,'val_loss':v_loss /total,'v_accuracy':correct / total})
            torch.save(encoder.state_dict(),output / f"epoch_{i}.pth")
            wb.save(output / f'epoch_{i}.pth')

