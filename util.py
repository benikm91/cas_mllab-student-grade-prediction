import pandas as pd
import os

def get_preprocessed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), path)
    )

def get_preprocessed_train_data() -> pd.DataFrame:
    return get_preprocessed_data('student-mat-train.csv')
