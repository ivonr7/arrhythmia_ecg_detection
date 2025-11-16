import numpy as np
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
            ),wb.rdrecord(str(Path(data_folder) / patient))


def extract_features(patients:pd.DataFrame,data_folder:str):
    annotations = []
    for annot,signal in tqdm(read_annots(data_folder,patients), desc="Reading Patient Annotations"):
        labels = annot.symbol
        indicies = annot.sample.tolist()
        name = repeat(annot.record_name,len(labels))
        freq = repeat(annot.fs,len(labels))
        note = pd.Series(annot.aux_note).fillna("")
        size = repeat(signal.sig_len,len(labels))
        annotations = chain(annotations,zip(name,labels,note,indicies,size,freq))
    return annotations



def main(adults:str,children:str,test_set:list, data_folder:str, out_folder:str):
    patients = join_patients(
        pd.read_csv(adults),
        pd.read_csv(children)
    )
    test_set = set(test_set)
    patient_mask = np.array([
        False if p in test_set else True for p in patients['file_name']
    ])
    print(patient_mask)
    train_data = extract_features(patients.loc[patient_mask,:],data_folder)
    test_data = extract_features(patients.loc[~patient_mask,:], data_folder)    
    # Concatenate the data
    # Write to output folder
    train_df = pd.DataFrame(
        data=list(train_data),
        columns=['file_name','labels','aux_note','indicies','length','frequency']
    )
    test_df = pd.DataFrame(
        data=list(test_data),
        columns=['file_name','labels','aux_note','indicies','length','frequency']
    )

    out = Path(out_folder)
    train = out / 'train'
    test = out / 'test'
    train.mkdir(exist_ok=True,parents=True)
    test.mkdir(exist_ok=True,parents=True)
    train_df.to_csv(
        train / 'patient_annotations.csv',
        index = False
    )
    test_df.to_csv(
        test / 'patient_annotations.csv',
        index = False
    )
    patients.to_csv(
        out / 'patient_metadata.csv'
    )

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--adult-meta',type=str,required=True, help='path to adult metadata file')
    parser.add_argument('--child-meta', type=str, required=True, help="path to child metadata csv")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to dataset(folder)")
    parser.add_argument('--out-dir',type=str, default="combined_dataset", help="Folder to write the metadata,signal and annotation files")
    args = parser.parse_args()
    test_set = ['x001','x006','x007','x108','x105','x026']
    main(
        args.adult_meta,
        args.child_meta,
        test_set,
        args.data_dir,
        args.out_dir
    )