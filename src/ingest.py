import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)