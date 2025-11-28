import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    sys.path.append("../arrhythmia_ecg_detection/")
    from src.pytorch_datasets.annot_dataset import Annot_Dataset, NORMAL_DICT
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    import marimo as mo
    return Annot_Dataset, NORMAL_DICT, mo, plt, sns


@app.cell
def _(sns):
    sns.set_theme(context="poster",style="white")
    return


@app.cell
def _(mo):
    split = mo.ui.radio(['Train','Test'],value="Train")
    split
    return (split,)


@app.cell
def _(Annot_Dataset, split):
    ds = Annot_Dataset(
        dataset_folder="physionet.org/files/leipzig-heart-center-ecg/1.0.0/",
        extracted_folder="dataset/",
        classification_task='binary',train=True if split.value == "Train" else False
    )
    return (ds,)


@app.cell
def _(NORMAL_DICT):
    def isnormal(beat):
        if beat in NORMAL_DICT:
            return "Arrythmia"
        return "Normal"
    return (isnormal,)


@app.cell
def _(ds, isnormal):
    classes = ds.beats['combined_label'].apply(isnormal).value_counts()
    return (classes,)


@app.cell
def _(classes, plt, sns, split):
    sns.barplot(classes)
    sns.despine()
    plt.title(f"{split.value} Label Distribution")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
