from prefect import task
import time
import numpy as np

class Result:
    def __init__(self, x: np.ndarray, fun: float):
        self.x = x
        self.fun = fun


@task(name="fit gamma sdp")
def fit_gamma_sdp(objective_fn: callable, initial_params: dict, options: dict):
    """
    gamma sdpを実行する
    """
    print("fit gamma sdp", objective_fn, initial_params, options)
    return Result(
        x=np.array([1, 2, 3]),
        fun=10
    )

