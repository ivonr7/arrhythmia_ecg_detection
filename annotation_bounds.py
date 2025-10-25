import pandas as pd
import numpy as np
import csv

ecg_path = "patients_ecg.csv"
annotations_path = "patient_annotations.csv"
output_path = "ecg_windows.csv"
window_size = 294  # ±0.3s

# Load signals
ecg_df = pd.read_csv(ecg_path)
ecg_df = ecg_df.dropna(axis=1, how='all')  # removes columns that are completely NaN

annotations_df = pd.read_csv(annotations_path)

signal_columns = [c for c in ecg_df.columns if c != 'time_s']
signals = ecg_df[signal_columns].values
time_s = ecg_df['time_s'].values

# Open CSV for writing
with open(output_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(['sample', 'time_s'] + signal_columns + ['symbol', 'aux_note'])
    
    for i, ann in enumerate(annotations_df.itertuples()):
        sample_idx = int(ann.sample)
        label_symbol = ann.symbol
        label_aux = ann.aux_note
        start = max(0, sample_idx - window_size)
        end = min(len(signals), sample_idx + window_size)
        
        window_signals = signals[start:end, :]
        window_times = time_s[start:end]
        
        #writing to csv
        for t_idx in range(len(window_signals)):
            row = [sample_idx, window_times[t_idx]] + window_signals[t_idx].tolist() + [label_symbol, label_aux]
            writer.writerow(row)

        
        if i % 50 == 0:
            print(f"Processed {i}/{len(annotations_df)} annotations")

print(f"CSV saved to {output_path}")
