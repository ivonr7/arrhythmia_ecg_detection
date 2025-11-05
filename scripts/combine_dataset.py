import pandas as pd
from itertools import repeat
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser

'''
    Join Adult and child datasets
'''
def join_patients(adults:pd.DataFrame, children:pd.DataFrame):
    adults['patient_type'] = list(repeat("adult",adults.shape[0]))
    children['patient_type']= list(repeat("child",children.shape[0]))
    return pd.concat([adults,children],axis=0, ignore_index=True).set_index("subject_id")

def read_signals(data_folder:str, patients:pd.DataFrame,ext:str = '.csv.gz'):
    for patient in patients['file_name']:
        yield (
            pd.read_csv(
                Path(data_folder) / (patient+ "_ecg" + ext)
            ),
            pd.read_csv(
                Path(data_folder) / (patient+ "_annotations" + ext)
            ),
        )

def main(adults:str,children:str, data_folder:str, out_folder:str):
    patients = join_patients(
        pd.read_csv(adults),
        pd.read_csv(children)
    )
    final_row = []
    signals = []
    annots = []
    for ecg, annot in tqdm(read_signals(data_folder, patients),desc=f"Combining Data"):
        final_row.append(ecg.shape[0] - 1)
        signals.append(ecg)
        annots.append(annot)

        
    
    # Concatenate the data
    signal_df = pd.concat(signals,axis=0,ignore_index=True)
    annot_df = pd.concat(annots, axis=0, ignore_index=True)
    patients['signal_end'] = final_row
    # Write to output folder
    out = Path(out_folder)
    out.mkdir(exist_ok=True,parents=True)
    signal_df.to_csv(
        out / "patients_ecg.csv.gz",
        compression='gzip'
    )
    annot_df.to_csv(
        out / 'patient_annotations.csv.gz',
        compression='gzip' 
    )
    patients.to_csv(
        out / 'patient_metadata.csv'
    )

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--adult-meta',type=str,required=True, help='path to addult metadata file')
    parser.add_argument('--child-meta', type=str, required=True, help="path to child metadata csv")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to exported csvs (folder)")
    parser.add_argument('--out-dir',type=str, default="combined_dataset", help="Folder to write the metadata,signal and annotation files")
    args = parser.parse_args()
    main(
        args.adult_meta,
        args.child_meta,
        args.data_dir,
        args.out_dir
    )