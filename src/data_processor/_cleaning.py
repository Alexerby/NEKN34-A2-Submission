import numpy as np
import pandas as pd
from src.utils import get_path


def _load_raw(source="ExchangeRate.csv"):
    path = get_path(source)
    # Read CSV and explicitly handle the date conversion without iloc assignment
    df = pd.read_csv(path)

    # Convert to datetime and immediately set as index to avoid dtype conflicts
    df.index = pd.to_datetime(df.iloc[:, 0])
    df = df.drop(df.columns[0], axis=1)  # Drop the original string column

    df.sort_index(inplace=True)

    # Return the first data column (the JPY/USD rate) as a Series
    return df.iloc[:, 0]


def get_dataset(id: str, transform="log", scale=100.0) -> pd.DataFrame | pd.Series:
    s = _load_raw()

    ranges = {
        "Dataset I": ("1978-01-03", "1994-06-29"),
        "Dataset II": ("1986-01-02", "2003-02-21"),
        "Extended": ("2003-01-01", "2023-12-31"),
    }

    # Add a global option to get everything for the overview plot
    if id == "Global":
        subset = s
    else:
        subset = s.loc[ranges[id][0] : ranges[id][1]]

    if transform == "log":
        return (np.log(subset / subset.shift(1)).dropna()) * scale

    return subset
