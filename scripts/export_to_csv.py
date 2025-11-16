#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import wfdb
import numpy as np
import pandas as pd

def find_record_basenames(data_dir: Path):
    """Return sorted unique basenames (without extension) that have a .hea file."""
    return sorted(p.stem for p in data_dir.glob("*.hea"))

def parse_channel_list(raw: str):
    # e.g., "I,II,V1" -> ["I","II","V1"]
    return [s.strip() for s in raw.split(",") if s.strip()]

def export_record_to_csv(data_dir: Path, record_name: str, seconds: float = 10.0,
                         channels=None, full: bool = False, out_dir: Path | None = None):
    rec_path = data_dir / record_name
    out_dir = out_dir or data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read header only to get fs and channel names (cheap)
    hdr = wfdb.rdheader(str(rec_path))
    fs = float(hdr.fs)
    all_ch = list(hdr.sig_name)

    # Decide which channels to export
    if channels:
        missing = [c for c in channels if c not in all_ch]
        if missing:
            raise ValueError(f"Requested channels not found in {record_name}: {missing}\n"
                             f"Available: {all_ch}")
        ch_idx = [all_ch.index(c) for c in channels]
        used_names = channels
    else:
        ch_idx = None
        used_names = all_ch

    # Decide the sample range to read
    if full:
        sampfrom = None
        sampto = None
        duration_info = "full record"
    else:
        win_len = int(round(seconds * fs))
        sampfrom = 0
        sampto = win_len
        duration_info = f"first {seconds:.3f} s"

    # Read signal slice as float (physical units)
    sig, fields = wfdb.rdsamp(str(rec_path), sampfrom=sampfrom, sampto=sampto, channels=ch_idx)
    n_samples = sig.shape[0]
    used_fs = float(fields.get("fs", fs))  # fs returned in fields too

    # Build time index (seconds)
    t = np.arange(n_samples, dtype=np.float64) / used_fs
    df = pd.DataFrame(sig, index=pd.Index(t, name="time_s"), columns=used_names)

    # Read matching annotations in the same range (if .atr exists)
    ann_csv_path = out_dir / f"{record_name}_annotations.csv"
    ecg_csv_path = out_dir / f"{record_name}_ecg.csv"

    try:
        ann = wfdb.rdann(str(rec_path), "atr", sampfrom=(sampfrom or 0), sampto=sampto)
        aux = getattr(ann, "aux_note", None) or getattr(ann, "aux", None)
        ann_df = pd.DataFrame({
            "sample": ann.sample,
            "time_s": ann.sample / used_fs,
            "symbol": ann.symbol,
            "aux_note": [a if a is not None else "" for a in (aux or [""] * len(ann.sample))]
        })
    except FileNotFoundError:
        ann_df = pd.DataFrame(columns=["sample", "time_s", "symbol", "aux_note"])

    # Save CSVs
    df.to_csv(ecg_csv_path)
    ann_df.to_csv(ann_csv_path, index=False)

    print(f"✔ {record_name}: wrote {duration_info} →")
    print(f"   ECG: {ecg_csv_path}")
    print(f"   ANN: {ann_csv_path}")
    print(f"   shape={df.shape}, fs={used_fs}, channels={list(df.columns)}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Export WFDB ECG signals and annotations to CSV."
    )
    parser.add_argument("--data-dir", default="data", help="Folder with WFDB files")
    parser.add_argument("--out-dir", default=None, help="Folder to write CSVs (default: data-dir)")
    parser.add_argument("--record", default=None,
                        help="Single record basename (e.g., x0017). If omitted, export all .hea basenames.")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="Seconds to export from start (ignored if --full). Default: 10")
    parser.add_argument("--full", action="store_true",
                        help="Export full record (warning: large memory/files).")
    parser.add_argument("--channels", default=None,
                        help="Comma-separated channel names to export (e.g., I,II,V1). Default: all")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    channels = parse_channel_list(args.channels) if args.channels else None

    if args.record:
        basenames = [args.record]
    else:
        basenames = find_record_basenames(data_dir)
        if not basenames:
            print(f"No .hea files found in {data_dir}")
            return

    for rec in basenames:
        try:
            export_record_to_csv(
                data_dir=data_dir,
                record_name=rec,
                seconds=args.seconds,
                channels=channels,
                full=args.full,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"✖ {rec}: {e}\n")

if __name__ == "__main__":
    main()
