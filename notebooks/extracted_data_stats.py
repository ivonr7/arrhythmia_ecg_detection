import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(f"""# Basic Statistics of Annotations""")
    return


@app.cell
def _():
    annot = pd.read_csv("/home/isaac/dev/sfu/cmpt310/arrhythmia_ecg_detection/combined_dataset/patient_annotations.csv.gz")
    annot.set_index('time_s')
    return (annot,)


@app.cell
def _(annot):
    annot.describe()
    return


@app.cell
def _(annot):
    annot.isna().sum(axis=0)
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(f"""# Basic Statistics of Signals""")
    return


@app.cell
def _():
    signal = pl.scan_csv("/home/isaac/dev/sfu/cmpt310/arrhythmia_ecg_detection/combined_dataset/patients_ecg.csv.gz")
    signal
    return (signal,)


@app.cell
def _(signal):
    signal.describe()
    return


@app.cell
def _(signal):
    signal.columns
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The raw ecg signals have more than 12 columns.  I, II, III, aVR,aVL, aVF and V1-6 have no NAN values while the rest do?  What do those values correspond to?""")
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(f"""# Plot Signal Values""")
    return


@app.function
def get_leads(df:pl.LazyFrame):
    return df.select(
        [
            "time_s","I","II","III",
            "aVR","aVL","aVF",
            "V1","V2","V3",
            "V4","V5","V6"
        ]
    )


@app.cell
def _(signal):
    leads = get_leads(signal)
    leads
    return (leads,)


@app.cell
def _(leads):
    leads.sort("time_s").rolling("time_s", period="1s").mean().collect()
    return


if __name__ == "__main__":
    app.run()
