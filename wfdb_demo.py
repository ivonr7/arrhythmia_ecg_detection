import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import wfdb as wdb
    import matplotlib.pyplot as plt
    return Path, plt, wdb


@app.cell
def _(Path):
    par_dir = Path("/home/isaac/dev/sfu/cmpt310/arrhythmia_ecg_detection/physionet.org/files/leipzig-heart-center-ecg/1.0.0")
    return (par_dir,)


@app.cell
def _():
    patient = "x001"
    return (patient,)


@app.function
def find_patient(dir:str,patient_id:str):
    exts = [".atr",".hea",".dat"]
    return [dir / (patient_id + ext) for ext in exts]


@app.cell
def _(par_dir, patient):
    files = find_patient(par_dir,patient)
    files[0]
    return


@app.cell
def _(par_dir, patient, wdb):
    header = wdb.rdheader(par_dir / patient)
    signal = wdb.rdrecord(par_dir / patient)
    annot = wdb.rdann(str(par_dir / patient), extension="atr")
    return annot, signal


@app.cell
def _(annot, plt, signal, wdb):
    wdb.plot_wfdb(record=signal, annotation=annot)
    plt.tight_layout()
    return


if __name__ == "__main__":
    app.run()
