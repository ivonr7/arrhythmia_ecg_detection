import pandas as pd
from itertools import repeat
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
import wfdb as wb
from itertools import chain
'''
    Join Adult and child datasets
'''
def join_patients(adults:pd.DataFrame, children:pd.DataFrame):
    adults['patient_type'] = list(repeat("adult",adults.shape[0]))
    children['patient_type']= list(repeat("child",children.shape[0]))
    return pd.concat([adults,children],axis=0, ignore_index=True).set_index("subject_id")

def read_annots(data_folder:str, patients:pd.DataFrame,ext:str = 'atr'):
    for patient in patients['file_name']:
        yield wb.rdann(
                str(Path(data_folder) / patient),ext
            )

def main(adults:str,children:str, data_folder:str, out_folder:str):
    patients = join_patients(
        pd.read_csv(adults),
        pd.read_csv(children)
    )
    annotations = []
    for annot in tqdm(read_annots(data_folder,patients), desc="Reading Patient Annotations"):
        labels = annot.symbol
        indicies = annot.sample.tolist()
        name = repeat(annot.record_name,len(labels))
        freq = repeat(annot.fs,len(labels))
        note = pd.Series(annot.aux_note).fillna("")
        annotations = chain(annotations,zip(name,labels,note,indicies,freq))
    
    # Concatenate the data
    # Write to output folder
    annot_df = pd.DataFrame(
        data=list(annotations),
        columns=['file_name','labels','aux_note','indicies','frequency']
    )
    out = Path(out_folder)
    out.mkdir(exist_ok=True,parents=True)

    annot_df.to_csv(
        out / 'patient_annotations.csv',
        index = False
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