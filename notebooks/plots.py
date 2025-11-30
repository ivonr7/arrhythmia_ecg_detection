import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import sys
    sys.path.append("../arrhythmia_ecg_detection/")
    from src.pytorch_datasets.annot_dataset import Annot_Dataset,NORMAL_DICT
    from src.models.cnn_extractor import CNN_Encoder
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import torch
    from itertools import chain
    from tqdm.auto import tqdm
    from sklearn.decomposition import PCA
    from random import sample
    return (
        Annot_Dataset,
        CNN_Encoder,
        NORMAL_DICT,
        PCA,
        mo,
        np,
        pd,
        plt,
        sample,
        sns,
        torch,
        tqdm,
    )


@app.cell
def _(Annot_Dataset):
    extracted = "dataset"
    data_folder = "physionet.org/files/leipzig-heart-center-ecg/1.0.0/"
    ds = Annot_Dataset(
        extracted_folder=extracted,
        dataset_folder=data_folder,
        window_duration=0.3
    )
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    n_samples = mo.ui.slider(start = 5000, stop = 100_000,label="How Many Samples")
    n_samples
    return (n_samples,)


@app.cell
def _(CNN_Encoder, torch):
    encoder_weights = torch.load("wandb/run-20251117_110331-i4zce57f/files/epoch_90.pth")
    encoder = CNN_Encoder(in_channels=1)
    encoder.load_state_dict(encoder_weights)
    encoder.eval()
    return (encoder,)


@app.cell
def _(encoder):
    def encode_beat(signal):
        y = encoder(signal.float().unsqueeze(0).unsqueeze(0))
        return y.detach().numpy().flatten()
    return (encode_beat,)


@app.cell
def _(ds, n_samples, sample, tqdm):
    idx = sample(range(2000,len(ds)), k=n_samples.value)
    windows = list([ds[i] for i in tqdm(idx)])
    return (windows,)


@app.cell
def _(encode_beat, np, tqdm, windows):
    signal = np.array([encode_beat(window[0]) for window in tqdm(windows)])
    label = np.array([window[1] for window in windows])
    return label, signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""# Plot Signals in 2D""")
    return


@app.cell
def _(sns):
    sns.set_theme(
        context='poster',
        style='whitegrid'
    )
    return


@app.cell
def _(PCA, label, pd, signal):
    pca = PCA(n_components=2)
    beats = pca.fit_transform(signal)
    beats = pd.DataFrame(data=beats, columns=[f'PC{i}' for i in range(2)])
    beats['labels'] = ['normal' if l == 0 else 'arrythmia' for l in label]
    return (beats,)


@app.cell
def _(beats, plt, sns):
    sns.scatterplot(beats, x = 'PC0', y = 'PC1',hue = 'labels',palette=['#fc8d62','#8da0cb'],alpha = 0.8)
    plt.legend(bbox_to_anchor = (1,1.2), ncols = 2,frameon = False)
    sns.despine()
    plt.show()
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(f"""# Plot Label Distribution""")
    return


@app.cell
def _(NORMAL_DICT):
    def isNormal(beat):
        return beat in NORMAL_DICT
    return (isNormal,)


@app.cell
def _(ds, isNormal):
    ds.beats['label'] = ds.beats['combined_label'].apply(isNormal)
    ds.beats
    return


@app.cell
def _(ds, plt, sns):
    label_cts = ds.beats['label'].value_counts()
    sns.barplot(label_cts)
    plt.xlabel('Is Arrythmia?')
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(f"""# Labels Per-Patient""")
    return


@app.cell
def _(ds, pd):
    patient_labels = pd.crosstab(ds.beats['file_name'],ds.beats['label']).rename(
        columns={False:'normal', True:'arrythmia'}
    )
    patient_labels
    return (patient_labels,)


@app.cell
def _(patient_labels):
    patient_probs = patient_labels.div(patient_labels.sum(axis=1),axis = 0)
    return (patient_probs,)


@app.cell
def _(patient_probs, plt, sns):
    plt.figure(figsize=(20,20))
    patient_probs.plot(kind = 'bar',stacked = True,figsize = (15,5))
    plt.legend(frameon = False,bbox_to_anchor = (0.3,1.1),ncols =2)
    plt.grid(False)
    plt.xlabel('Patient')
    sns.despine()
    plt.show()
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(f"""# Samples per-patient""")
    return


@app.cell
def _(ds, plt):
    annot_counts = ds.beats.groupby('file_name').count().min(axis = 1).sort_values(ascending = False)
    annot_counts.plot(kind = 'bar',figsize = (15,5))
    plt.ylabel("Annotation Count")
    plt.xlabel('Patient')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
