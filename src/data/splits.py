"""
Section 7 — Train / validation / test split.

"""
from sklearn.model_selection import train_test_split
from config import TRAIN_SIZE, TEST_SIZE, VAL_SIZE



def split_data(df, label_column="encoded_status", random_state=42):
    """
    Split df into train (60%), val (20%), test (20%).
    """
    # First split: split train for 60%, save remaining inside temp %40 for val and test.
    train_df, temp_df = train_test_split(df,
                                         train_size=TRAIN_SIZE,
                                         stratify=df[label_column],
                                         random_state=random_state,
                                         )

    # Second split: divide temp 50 - 50 into val and test sets.
    # val_size / (val_size + test_size) = 0.5 here
    # Written this way to enable easy experimentation with different train / val / test ratios
    val_df, test_df = train_test_split(temp_df,
                                       train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE),
                                       stratify=temp_df[label_column],
                                       random_state=random_state,
                                       )
    # resetting indexes to prevent confusion as they carry their old indexes from the initial dataframe
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # This sanity check  , checking shapes and class balance across all three splits.
    print(f"train: {train_df.shape}, {len(train_df)/len(df):.3f} of total")
    print(f"val:   {val_df.shape}, {len(val_df)/len(df):.3f} of total")
    print(f"test:  {test_df.shape}, {len(test_df)/len(df):.3f} of total")
    print("*"*100)
    print("train class distribution:")
    print(train_df[label_column].value_counts(normalize=True).sort_index())
    print("*"*100)
    print("val class distribution:")
    print(val_df[label_column].value_counts(normalize=True).sort_index())
    print("*"*100)
    print("test class distribution:")
    print(test_df[label_column].value_counts(normalize=True).sort_index())

    return train_df, val_df, test_df

if __name__ == "__main__":
    # run python -m src.data.splits to run sanity check  
    from pathlib import Path
    import pandas as pd
    df_file_path = Path().cwd() / "preprocessed_df.csv"
    if df_file_path.exists():
        df = pd.read_csv(df_file_path)

        train_df, val_df, test_df = split_data(df)
    else:
        print("dataframe file does not exist, please open 01_pipeline.ipynb and run through section 7 to create the dataframe csv file")