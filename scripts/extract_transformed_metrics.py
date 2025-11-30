import wandb as wb
import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("download arrformer training metrics")
    parser.add_argument("run", type=str, help='run id from weights and biases under overview tab')
    parser.add_argument("out", type = str, help='where to write file and what to name it')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    api = wb.Api()
    run = api.run(args.run)
    history = run.history()
    metrics = pd.DataFrame()
    metrics['train_loss'] = history['avg_loss']
    metrics['val_loss'] = history['val_loss']
    metrics['val_acc'] = history['v_accuracy']
    metrics['epoch'] = history['epoch']
    clean = (
        metrics.groupby('epoch')
        .agg({
           'train_loss': 'max',
           'val_loss': 'max',
           'val_acc': 'max'
        })
    )
    print(clean)
    clean.to_csv(args.out)