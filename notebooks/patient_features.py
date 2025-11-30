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
    from child_distribution import fix_age
    import marimo as mo


@app.cell
def _():
    sns.set_theme(
        context='poster',
        style='whitegrid'
    )
    return


@app.cell
def _():
    mo.md(f"""
    # Gender Ratio
    """)
    return


@app.cell
def _():
    patients = pd.read_csv("dataset/patient_metadata.csv").set_index('file_name').pipe(fix_age)
    patients
    return (patients,)


@app.cell
def _(patients):
    gender_ratio = patients.groupby('gender').count()

    sns.barplot(gender_ratio, x = 'gender', y = 'subject_id')
    plt.ylabel("Number of Patients")
    return


@app.cell
def _(patients):
    age_class = patients.groupby('patient_type').count()

    sns.barplot(age_class, x = 'patient_type', y = 'subject_id')
    plt.ylabel("Number of Patients")
    plt.xlabel('Children vs Adults')

    return


if __name__ == "__main__":
    app.run()
