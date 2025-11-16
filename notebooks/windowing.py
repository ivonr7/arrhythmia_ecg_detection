import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd
    import numpy as np
    import wfdb as wb
    import polars as pl
    import marimo as mo
    import sys
    from pathlib import Path



@app.cell
def _():
    index = mo.ui.file_browser()
    return (index,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Load Metadata File
    select metadata file to read.  Only the first file will be read
    """
    )
    return


@app.cell(hide_code=True)
def _(index):
    index
    return


@app.cell
def _(index):

    file = index.value[0].id
    file = Path(file)
    folder = file.parent
    return file, folder


@app.function
def to_timestamp(df):
    df["ecg_duration"] = pd.to_timedelta(df["ecg_duration"]) 
    return df


@app.cell
def _(file):
    patient_meta = pd.read_csv(file)
    patient_meta = patient_meta.pipe(to_timestamp).set_index('subject_id')
    patient_meta
    return (patient_meta,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""Add File Paths to metadata file.  Later useful to read patient data""")
    return


@app.function
def compute_file(patient_id:str, folder:str, ext:str = '.atr'):
    file = Path(folder) / patient_id
    if ext == '.atr' or ext == '.hea' or ext == '.dat':
        return file
    raise ValueError(f"Illegal Extension {ext} only, (.hea,.dat,.atr) supported")


@app.cell
def _(folder, patient_meta):
    patient_meta['atr'] = patient_meta['file_name'].apply(compute_file, folder = folder, ext = '.atr')
    patient_meta['hea'] = patient_meta['file_name'].apply(compute_file, folder = folder, ext = '.hea')
    patient_meta['dat'] = patient_meta['file_name'].apply(compute_file, folder = folder, ext = '.dat')
    return


@app.cell
def _(patient_meta):
    patient_meta
    return


@app.cell
def _(patient_meta):
    patient = mo.ui.dropdown(patient_meta.index,100, label="Choose Patient")
    patient
    return (patient,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""An example of Reading an annotation file. Showing key attibutes""")
    return


@app.cell
def _(patient, patient_meta, start, width):
    annot = wb.rdann(str(patient_meta.at[patient.value,'atr']),'atr',sampfrom=start.value,sampto=start.value + width.value, shift_samps=True)
    annot.symbol,annot.sample
    return (annot,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""An example of reading an annotation file.  The annotation symbols are stored in the ```.symbol``` attribute while the signal indexes are stored in the ```.sample``` attribute""")
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(
        r"""
    ## Reading Record Files
    record files for wfdb are where the signal is stored.  Windows of the files can be read directly.  using the ```sampfrom``` and ```sampto``` keyword arguments to the ```wb.rdrecord``` command
    """
    )
    return


@app.cell
def _():
    start = mo.ui.slider(start=0,stop=100_000,label="Start")
    width = mo.ui.slider(start = 10, stop=10_000, label="Window Size")
    mo.vstack([start,width])
    return start, width


@app.cell
def _(patient, patient_meta, start, width):
    sig = wb.rdrecord(str(patient_meta.at[patient.value,'atr']),sampfrom=start.value, sampto=start.value + width.value)
    return (sig,)


@app.cell(hide_code=True)
def _(sig):
    freq = sig.fs
    mo.md(f"""
    Record {sig.record_name}: \n
    - Sampling Frequency {sig.fs} (hz)
    - Number of Samples {sig.sig_len}
    - Start time {sig.base_datetime}
    - Channels Bit Depth: {sig.fmt}
    - Window Duration: {1/sig.fs * sig.sig_len:.4f} (s)
    - Comments\n
    {sig.comments}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""For WFDB [Records]("https://wfdb.readthedocs.io/en/latest/wfdb.html"), either the ```p_signal``` or ```d_signal``` are set and refer to the actual samples. These can be used to extract numpy arrays of samples""")
    return


@app.cell
def _(sig):
    sig.p_signal[10,:], sig.d_signal,sig.e_p_signal,sig.e_d_signal
    return


@app.cell
def _(annot, sig):
    wb.plot_wfdb(record=sig, annotation=annot,figsize=(8,20))

    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""## Use Annotation to Index Record""")
    return


@app.cell
def _(patient, patient_meta):
    test_annot = wb.rdann(str(patient_meta.at[patient.value,'atr']),'atr')
    test_annot
    return (test_annot,)


@app.cell
def _(test_annot):
    a = test_annot.symbol[100]
    i = test_annot.sample[100]
    a,i
    return


@app.function
def compute_window(record:str,index:int,window_size:int = 294):
    """
        This Function Extracts the signal window around a an annotation 
        building off yasnas code. If the signal window is OOB pad with zero to 
        retain window size
    """
    signal_min = index - window_size
    signal_max = index + window_size
    padding = 0
    if signal_min < 0:
        padding= abs(signal_min)
        signal_min = 0
    record = wb.rdrecord(record,sampfrom=signal_min,sampto=signal_max)
    return np.pad(record.p_signal,((padding,0), (0,0)))


@app.cell
def _(patient, patient_meta):
    window = compute_window(record=str(patient_meta.at[patient.value,'atr']),index=0,window_size=5)
    window.shape
    return


if __name__ == "__main__":
    app.run()
