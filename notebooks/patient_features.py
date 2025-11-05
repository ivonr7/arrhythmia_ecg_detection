import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd
    import numpy as np
    from patient_extraction import extract_features
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell
def _():
    patients = pd.read_csv("dataset/patient_metadata.csv").set_index('file_name')
    patients
    return (patients,)


@app.cell
def _():
    annotations = pd.read_csv("dataset/patient_annotations.csv")
    annotations
    return (annotations,)


@app.cell
def _(annotations):
    beats = annotations[['file_name','labels']].groupby('labels').count().rename(
        columns={"file_name":"Count"}).sort_values(by="Count",ascending=False)
    sns.barplot(beats.T)
    plt.title("Count Type of Beats In Dataset")
    return


@app.cell
def _(annotations):
    plt.figure(figsize=(10,10))
    rythmns = annotations[['file_name','aux_note']].groupby('aux_note').count().rename(
        columns={"file_name":"Count"}).sort_values(by="Count",ascending=False)
    sns.barplot(rythmns.T,)
    plt.xticks(rotation = 45)
    plt.title("Counts of Rythmn Annotations")
    return


@app.cell
def _():
    normal_beats = ['N','R','L','b','j']
    return (normal_beats,)


@app.cell
def _(normal_beats):
    def is_normal(row):
        normals = row[normal_beats].sum()
        total = row.sum()
        return normals , (total-normals)
    return (is_normal,)


@app.cell
def _(annotations):
    count_per_class = pd.crosstab(annotations['file_name'], annotations['labels'])
    count_per_class
    return (count_per_class,)


@app.cell
def _(count_per_class, is_normal):
    bin_counts = count_per_class.apply(is_normal, axis = 1,result_type='expand').rename(
        columns={0:'Normal',1:"Arrythmia"})
    bin_counts
    return (bin_counts,)


@app.function
def to_prop(row):
    total = row.sum()
    return row['Normal'] / total, row['Arrythmia'] / total


@app.cell
def _(bin_counts):
    bin_prop = bin_counts.apply(to_prop,axis = 1,result_type='expand').rename(
        columns={0:'Normal',1:"Arrythmia"})
    bin_prop
    return (bin_prop,)


@app.cell
def _(bin_prop):
    bin_prop.plot(kind = 'bar',stacked = True)
    plt.legend(frameon = False,bbox_to_anchor = (1,1))
    plt.title("Proportion of Normal Beats Per-Patient")
    return


@app.cell
def _(bin_prop):
    sns.barplot(bin_prop.sum(axis = 0) / 39)
    plt.ylabel("Proportion")
    plt.title("Proportion of Normal Beats vs Arrythmia in Dataset")
    return


@app.cell
def _(bin_counts, patients):
    patients['abnormal'] = bin_counts['Arrythmia']
    patients
    return


@app.cell
def _(patients):
    abnormals = patients['abnormal'].sort_values(ascending=False).cumsum()/ patients['abnormal'].sum()
    abnormals
    return (abnormals,)


@app.cell
def _(patients):
    sns.barplot(patients['abnormal'].sort_values(ascending=False) / patients['abnormal'].sum())
    plt.xticks(rotation = 90)
    plt.title("Percentage Of Abnormal Beats Per-Patient")
    return


@app.cell
def _(abnormals):
    sns.lineplot(abnormals)
    plt.xticks(rotation = 90)
    plt.title("Cumulative Number of Abnormal Beats")
    return


if __name__ == "__main__":
    app.run()
