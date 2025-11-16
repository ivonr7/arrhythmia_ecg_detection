import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import sys
    sys.path.append("../arrhythmia_ecg_detection/")
    from src.pytorch_datasets.annot_dataset import Annot_Dataset
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.decomposition import IncrementalPCA, PCA

    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.spatial.distance import pdist


@app.cell
def _():
    extracted = "dataset"
    data_folder = "physionet.org/files/leipzig-heart-center-ecg/1.0.0/"
    ds = Annot_Dataset(
        extracted_folder=extracted,
        dataset_folder=data_folder,
        window_duration=0.3
    )
    return (ds,)


@app.cell
def _(ds):
    ds[5000][0].shape
    return


@app.function
def chain_windows(ds:Annot_Dataset):
    for i in tqdm(range(2000,20_000)):
        yield ds[i][0].numpy().mean(axis = 0)


@app.cell
def _(ds):
    signal = np.array(list(chain_windows(ds)))
    signal = signal.reshape(-1,12)
    return (signal,)


@app.cell
def _(signal):
    pca = PCA()
    points = pca.fit_transform(signal)
    viz = np.random.choice(np.arange(points.shape[0]), size=100_000)
    plt.scatter(points[:,0],points[:,1])
    return pca, points


@app.cell
def _(pca):
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("PC")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance of PC's")
    return


@app.cell
def _(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), label = "Cumulative Explained Variance")
    plt.axvline(x = 3,c = 'red', label = '90% Explained Variance')
    plt.xlabel("Number of PCs")
    plt.ylabel("Explained Variance")
    plt.legend(frameon = False,bbox_to_anchor = (1,1.1),ncols = 2)
    plt.grid(which='major')
    plt.show()
    return


@app.cell(column=1)
def _(ds):
    y = ds.annot['labels']
    return (y,)


@app.cell
def _(y):
    y.index
    return


@app.cell
def _(y):
    labels = y[2000:20_000]
    labels.value_counts().plot.bar()
    return (labels,)


@app.cell
def _(labels, points):
    n = (points[:,:2] - points[:,:2].mean(axis = 0)) / points[:,:2].std(axis = 0)

    samples = pd.DataFrame(data = n, columns = ["PC1","PC2"])
    samples['annot'] = labels
    return (samples,)


@app.cell
def _(samples):
    values = samples.loc[np.logical_and(samples['PC1'] > -10,samples['PC2'] > -10)]
    return (values,)


@app.cell
def _(values):
    import seaborn as sns
    sns.scatterplot(values,x = 'PC1', y = 'PC2',hue='annot',alpha = 0.5)
    plt.legend(bbox_to_anchor = (1,1))
    return (sns,)


app._unparsable_cell(
    r"""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    pts = tsne.fit_transform(signal)
    plt.scatter(pts[:,0],pts[:,1],alpha = 0.2,c = )
    """,
    name="_"
)


@app.cell
def _(labels, pts, sns):
    ts_pts = pd.DataFrame(data=pts,columns=['T1','T2'])
    ts_pts['y'] = labels
    sns.scatterplot(ts_pts,x = 'T1',y='T2',hue = 'y', alpha = 0.3)
    return (ts_pts,)


@app.cell(column=2)
def _(sns, ts_pts):
    g = sns.FacetGrid(ts_pts, row = 'y')
    g.map(sns.scatterplot, "T1","T2",alpha = 0.5)
    return


@app.cell
def _(labels, signal):
    from sklearn.feature_selection import mutual_info_classif
    score = mutual_info_classif(signal, labels)
    score = pd.Series(data=score, index=np.arange(12)).sort_values()
    score
    return (score,)


@app.cell
def _(score):
    score.plot.bar()
    plt.ylabel("Mutual Information")
    plt.title("ECG Channel MI with Annotations")
    return


@app.cell(column=3)
def _(signal):
    c = np.corrcoef(signal,rowvar=False)
    c.shape
    return (c,)


@app.cell
def _(signal):
    signal.shape
    return


@app.cell
def _(c, sns):
    sns.heatmap(np.abs(c),annot=True,fmt=".1f")
    plt.title("Correlation Between ECG Channels")
    return


@app.cell(column=4)
def _():
    return


@app.cell
def _(ds):
    import marimo as mo
    channel = mo.ui.dropdown(label = "ECG Channel",options = np.arange(12).tolist(),value = 0)
    index = mo.ui.slider(start=2000, stop=len(ds) - 1)
    channel, index
    return channel, index


@app.cell
def _(channel, ds, index):
    sample = ds[index.value][0].numpy()
    frequencies = np.abs(np.fft.fftshift(np.fft.fft(sample[:,channel.value])))
    plt.plot(frequencies)
    return (sample,)


@app.cell
def _(channel, sample):
    plt.plot(sample[:,channel.value])
    return


if __name__ == "__main__":
    app.run()
