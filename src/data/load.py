"""
Section 3 — Dataset loading and label encoding.
This module is responsible for turning the raw CSV into a clean DataFrame
with an encoded label column. 
However, this module is not responsible from text cleaning. 
Text cleaning is done in preprocessing module (src/preprocessing)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import  RAW_DATA_PATH

def load_raw_data(path):
    """
    Load Combined_Data.csv and return a cleaned DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of Combined Data.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: 'statement' (raw text), 'status' (raw label string).

    following operations are done;
    1- unnamed index column is dropped
    2- Null and NaN ' are dropped.

    """
    df = pd.read_csv(path)
    df.drop("Unnamed: 0",axis = 1,inplace=True)
    df.dropna(inplace=True)
    assert sum(df.isna().sum()) == 0
    return df
def encode_labels(df, label_column="status"):
    """
    Fit a LabelEncoder on the label column and add an 'encoded_status' column.
    Parameters
    ----------
        df : pandas.DataFrame
        label_column : str or None

    Returns
    -------
    pandas.DataFrame, sklearn.preprocessing.LabelEncoder

    
    """
    labelencoder = LabelEncoder()
    labelencoder.fit(df[label_column])
    df["encoded_status"] = labelencoder.transform(df[label_column])
    return df, labelencoder
