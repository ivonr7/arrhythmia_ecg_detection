#pre-processing
from child_distribution import clean
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def onehot(df:pd.DataFrame,features: list = []):
    categorical = df[features]
    one = OneHotEncoder(sparse_output=False)
    one.fit(categorical)
    feats = one.transform(X = categorical)
    return pd.DataFrame(
        data=one.transform(X = categorical),
        columns= one.get_feature_names_out()
    )

def standardize(df:pd.DataFrame, features: list = []):
    return (df[features] - df[features].mean(axis = 0)) / df[features].std()
def normalize_age(df:pd.DataFrame, features:list = []):
    return (df[features])/ 100 
def duration(df:pd.DataFrame, features:list = []):
    time = df[features].squeeze().dt.total_seconds()
    return time 
def nothing(df:pd.DataFrame, features:list = []):
    return df[features]

feature_extract = {
    "gender": onehot,
    "age": standardize,
    "diagnosis": onehot,
    "ap_location": onehot,
    "ecg_duration":duration 
}


def extract_features(df:pd.DataFrame, extraction:dict = feature_extract):
    cleaned = clean(df)
    extracted = []
    for feature, extraction in feature_extract.items():
        extracted.append(
            extraction(cleaned, features = [feature])
        )
    return pd.concat(extracted, axis=1)