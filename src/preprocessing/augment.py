"""
Section 8 — EDA (Easy Data Augmentation) on minority classes.

"""
import nlpaug.augmenter.word as naw
import pandas as pd
from typing import List

aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)

# aug_p=0.3 means roughly 30% of eligible words get replaced per call
# This is the default of aug_p which I found reasonable.
# If it is set too low and augmented copies would barely differ from the original.

def augment_minority_classes(train_df:pd.DataFrame, minority_classes:List[str], label_column="status",
                              text_column="statement_seq", target_count=None) -> pd.DataFrame:
    """
        Generate synonym augmented samples for the given minority classes and
        append them to train_df, bringing each up to target_count.

        Uses synonym replacement only.
        matching Shorten et al. (2021)'s EDA technique choice 

        Each minority class is sampled WITH replacement up to the number of
        rows needed (samples_needed = target_count - current class size),
        since most minority classes need more augmented rows than they have
        original sentences. Synonym replacement is then applied independently
        to each sampled row.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training data only 
        minority_classes : list[str]
            Exact label strings to augment (e.g. config.MINORITY_CLASSES_FOR_AUGMENTATION).
        label_column : str
        text_column : str
            Which text column to augment, default is 'statement_seq'
        target_count : int or None
            Row count each minority class is brought up to. If None, computed
            automatically as the smallest non minority class's count, so every
            minority class ends up roughly balanced against the majority
            classes without an arbitrarily chosen multiplier.

        Returns
        -------
        pandas.DataFrame
            train_df with augmented rows appended. 
        """
    counts = train_df[label_column].value_counts()
    if target_count is None:
        # get number of observation for the least non-minority class
        target_count = counts[~counts.index.isin(minority_classes)].min()

    augmented_rows = []
    for minority_class in minority_classes:
        class_subset = train_df[train_df[label_column] == minority_class]
        samples_needed = target_count - len(class_subset)
        samples = class_subset.sample(samples_needed, replace=True)
        for ind, row in samples.iterrows():
            augmented_text = aug.augment(row[text_column])
            augmented_row = row.copy()
            augmented_row[text_column] = augmented_text
            augmented_rows.append(augmented_row)

    augmented_df = pd.DataFrame(augmented_rows)
    train_df = pd.concat([train_df, augmented_df], ignore_index=True)

    print(f"before augmentation : \n")
    print(counts)
    print("*"*100)
    print(f"after augmentation : \n")
    print(train_df[label_column].value_counts())

    return train_df
