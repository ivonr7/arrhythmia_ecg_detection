import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder containing your ECG triplets
data_dir = os.path.join(os.getcwd(), "data")
record_name = "x106"   # change to the actual patient ID in your folder

# Load record (.hea + .dat)
record = wfdb.rdrecord(os.path.join(data_dir, record_name))
print("Loaded record:", record_name)
print("Signal type:", type(record.p_signal))
print("Signal shape:", record.p_signal.shape)
print("Channels:", record.sig_name)
print("Sampling rate (Hz):", record.fs)

# Load annotation (.atr)
ann = wfdb.rdann(os.path.join(data_dir, record_name), "atr")
print("Loaded annotation file")
print("Annotation type:", type(ann.sample))
print("Annotation dtype:", ann.sample.dtype)
print("Number of annotations:", len(ann.sample))

# Convert to pandas
t = np.arange(record.p_signal.shape[0]) / record.fs
df = pd.DataFrame(record.p_signal, index=pd.Index(t, name="time_s"), columns=record.sig_name)
print("\nPandas DataFrame preview:\n", df.head())

# Plot the first 5 seconds of Lead 1
plt.figure(figsize=(10, 4))
plt.plot(t[:int(5*record.fs)], record.p_signal[:int(5*record.fs), 0])
plt.xlabel("Time (s)")
plt.ylabel("mV")
plt.title(f"ECG Signal – {record_name} (Lead {record.sig_name[0]})")
plt.tight_layout()
plt.show()
