from sklearn.model_selection import train_test_split
import pandas as pd
from util import get_preprocessed_orig_data


def make_dirty(df: pd.DataFrame):
    def deterministic_suffle(df: pd.DataFrame):
       return df.sample(frac=1, random_state=42)
    df = deterministic_suffle(df)
    # A few "unknown" for features about family. We can interpolate them later
    fam_cols = ['famsize', 'famsup', 'famrel']
    df.loc[df.index[1:23], fam_cols] = None
    df = deterministic_suffle(df)
    # A lot "unknown" for reason, so we can simply drop reason feature later
    df.loc[df.index[1:311], ['reason']] = None
    df = deterministic_suffle(df)
    # -1 in absence for "unknown"
    df.loc[df.index[1:12], ['absences']] = -1
    return deterministic_suffle(df).reset_index(drop=True)


data = get_preprocessed_orig_data()

data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)

data_train = make_dirty(data_train)
data_test = data_test.reset_index(drop=True)

data_train.to_csv('../student-mat-train.csv')
data_train.to_csv('../student-mat-test.csv')