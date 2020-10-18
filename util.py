import pandas as pd
import os


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    def to_swiss_grade(grade, max_grade = 20):
        return grade / max_grade * 5 + 1
    df.loc[:, ['G1', 'G2', 'G3']] = df[['G1', 'G2', 'G3']].apply(to_swiss_grade)
    df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    return df


def get_preprocessed_data(path: str) -> pd.DataFrame:
    return preprocess(
        pd.read_csv(
            os.path.join(os.path.dirname(__file__), path)
        )
    )


def get_preprocessed_orig_data() -> pd.DataFrame:
    return get_preprocessed_data('student-mat-orig.csv')


def get_preprocessed_train_data() -> pd.DataFrame:
    return get_preprocessed_data('student-mat-train.csv')
