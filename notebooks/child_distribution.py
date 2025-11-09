import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import pandas as pd
    from pandas import to_timedelta
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Read Patient Info For children""")
    return


@app.cell
def _():
    patients = pd.read_csv("/home/isaac/dev/sfu/cmpt310/arrhythmia_ecg_detection/physionet.org/files/leipzig-heart-center-ecg/1.0.0/children-subject-info.csv")
    patients
    return (patients,)


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    # Clean Dataset
    * Fix Age column (.14.3)
    * fill na values in location --> unknown
    * change column name to ap_loacation --> ap_location
    """
    )
    return


@app.function
def rm_dot(col):
    row = str(col).strip(".")
    try:
        return float(row)
    except Exception() as e:
        return None


@app.function
def fix_age(df):
    df["age"] = df["age"].apply(rm_dot)
    return df


@app.function
def fill_location(df):
    df["ap_location"] = df['ap_loacation'].fillna("unknown")
    return df.drop(columns = "ap_loacation")


@app.function
def clean(df):
    return (
        df
        .pipe(fix_age)
        .pipe(fill_location)
        .pipe(to_timestamp)
    )


@app.function
def to_timestamp(df):
    df["ecg_duration"] = to_timedelta(df["ecg_duration"]) 
    return df


@app.cell
def _(patients):
    cleaned = clean(patients)
    cleaned.loc[1,"ecg_duration"]
    return (cleaned,)


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    ## Plot Patient Characteristics
    * barplot gender
    * histogram age (x-axis 0-18)
    * barplot diagnosis
    * cumulative plot for age (sort --> plot curve)
    * ap_location barplot
    """
    )
    return


@app.cell
def _():
    sns.set_theme()
    return


@app.cell
def _(cleaned):
    gender_ratio = cleaned["gender"].value_counts() / cleaned.shape[0] * 100
    sns.barplot(gender_ratio.reset_index(),x = "gender", y = "count")
    plt.ylabel("percentage")
    plt.title("Gender ratio for patients under 18")
    return


@app.cell
def _(cleaned):
    plot = sns.histplot(cleaned["age"], bins= 15)
    plt.xlim((0,18))
    plt.title("Distribution of Age for Patients under 18")
    plot
    return


@app.cell
def _(cleaned):
    diagnoses = cleaned["diagnosis"].value_counts() / cleaned.shape[0] * 100
    sns.barplot(diagnoses.reset_index(), x="diagnosis", y = "count")
    plt.ylabel("percentage")
    plt.title("Percentage of Patients with each Diagnosis")
    return


@app.cell
def _(cleaned):
    duration_per_patient = cleaned[["file_name","ecg_duration"]].sort_values("ecg_duration")
    cumulative_durations = duration_per_patient["ecg_duration"].diff().fillna(pd.Timedelta(0)).cumsum()
    seconds = cumulative_durations.dt.total_seconds().reset_index()
    seconds
    return duration_per_patient, seconds


@app.cell
def _(duration_per_patient, seconds):
    plt.figure(figsize=(10,5))
    cum = plt.plot(seconds["ecg_duration"])
    plt.xticks(np.arange(seconds.shape[0]),labels=duration_per_patient["file_name"], rotation = 45)
    plt.xlabel("Patient")
    plt.ylabel("ECG Duration (s)")
    plt.title("Cumulative ECG Duration Per Patient")
    cum
    return


if __name__ == "__main__":
    app.run()
