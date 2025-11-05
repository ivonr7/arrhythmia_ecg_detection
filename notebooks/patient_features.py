import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd
    import numpy as np
    from patient_extraction import extract_features


@app.cell
def _():
    df = patients = pd.read_csv("/home/isaac/dev/sfu/cmpt310/arrhythmia_ecg_detection/physionet.org/files/leipzig-heart-center-ecg/1.0.0/children-subject-info.csv")
    patients
    return (patients,)


@app.cell
def _(patients):
    extracted = extract_features(patients)
    return (extracted,)


@app.cell
def _(extracted):
    extracted
    return


if __name__ == "__main__":
    app.run()
