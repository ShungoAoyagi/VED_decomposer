from prefect import task
import numpy as np

@task(name="create objective function")
def create_objective_function(data: np.ndarray) -> callable:
    """
    目的関数を作成する
    """
    return lambda x: np.sum(x)