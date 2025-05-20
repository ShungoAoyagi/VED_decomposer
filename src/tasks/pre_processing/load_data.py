from prefect import task
import numpy as np

@task(name="load data")
def load_data(data_path: str) -> np.ndarray:
    """
    データを読み込む
    """
    return np.loadtxt(data_path)

