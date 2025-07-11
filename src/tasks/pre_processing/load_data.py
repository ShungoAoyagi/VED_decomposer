from prefect import task
import numpy as np
from src.tasks.pre_processing.settings import Settings
import math
import os
import json
from src.helpers.pick_partial_data import pick_partial_data

@task(name="load data")
def load_data(data_path: str, settings: Settings) -> np.ndarray[tuple[int, int, int], float]:
    """
    load xplor file
    set data to 0 if r > r_max
    """

    if os.path.exists("cache/data_modified.npy") and os.path.exists("cache/log.json"):
        log = json.load(open("cache/log.json", "r"))
        center_idx = log["center_idx"]
        lattice_params = log["lattice_params"]
        v = log["v"]
        # r_max = log["r_max"]
        max_idx = log["max_idx"]
        min_idx = log["min_idx"]
        settings.update_center_idx(center_idx)
        settings.update_lattice_params(lattice_params)
        settings.update_v(v)
        if max_idx is None or min_idx is None:
            tmp_max_idx = np.zeros(3, dtype=int)
            tmp_min_idx = np.zeros(3, dtype=int)
            for i in range(3):
                tmp_max_idx[i] = (center_idx[i] + int(settings.r_max * v[i] / lattice_params[i])) % v[i]
                tmp_min_idx[i] = (center_idx[i] - int(settings.r_max * v[i] / lattice_params[i])) % v[i]
            print(f"load_data DEBUG: tmp_max_idx: {tmp_max_idx}")
            print(f"load_data DEBUG: tmp_min_idx: {tmp_min_idx}")
            settings.update_max_idx(tmp_max_idx)
            settings.update_min_idx(tmp_min_idx)

        data = np.load("cache/data_modified.npy", allow_pickle=True)
        return pick_partial_data(data, settings)
    v = np.zeros(3, dtype=int)
    v_max = np.zeros(3, dtype=int)
    v_min = np.zeros(3, dtype=int)
    lattice_params = np.zeros(6, dtype=float)
    # load data efficiently by using generator
    with open(data_path, 'r') as f:
        # skip first 3 lines
        for _ in range(3):
            next(f)

        tmp = f.readline().split()
        v[0] = int(tmp[0])
        v_min[0] = int(tmp[1])
        v_max[0] = int(tmp[2])
        v[1] = int(tmp[3])
        v_min[1] = int(tmp[4])
        v_max[1] = int(tmp[5])
        v[2] = int(tmp[6])
        v_min[2] = int(tmp[7])
        v_max[2] = int(tmp[8])
        tmp_data = np.zeros((v[0], v[1], v[2]))
        settings.update_v(v)

        tmp = f.readline().split()
        lattice_params[0] = float(tmp[0])
        lattice_params[1] = float(tmp[1])
        lattice_params[2] = float(tmp[2])
        lattice_params[3] = float(tmp[3])
        lattice_params[4] = float(tmp[4])
        lattice_params[5] = float(tmp[5])
        settings.update_lattice_params(lattice_params)

        for _ in range(1):
            next(f)

        for i in range(v_min[2], v[2]):
            count = 0
            tmp = f.readline().split()
            for j in range(v_min[1], v[1]):
                for k in range(v_min[0], v[0]):
                    if (count % 5 == 0):
                        tmp = f.readline().split()
                    try:
                        tmp_data[k, j, i] = float(tmp[count % 5])
                    except:
                        print(tmp)
                        print(count)
                        print(k, j, i)
                        raise Exception("Error")
                    count += 1
    center = settings.center
    r_min = settings.r_min
    r_max = settings.r_max
    center_idx = [int(center[l] * v[l]) for l in range(3)]
    print(f"load_data DEBUG: center_idx: {center_idx}")
    settings.update_center_idx(center_idx)

    tmp_max_idx = np.zeros(3, dtype=int)
    tmp_min_idx = np.zeros(3, dtype=int)
    for i in range(3):
        tmp_max_idx[i] = (int((r_max / lattice_params[i] + center[i]) * v[i])) % v[i]
        tmp_min_idx[i] = (int((-r_max / lattice_params[i] + center[i]) * v[i])) % v[i]
    print(f"load_data DEBUG: tmp_max_idx: {tmp_max_idx}")
    print(f"load_data DEBUG: tmp_min_idx: {tmp_min_idx}")
    print(f"load_data DEBUG: r_max: {r_max}")
    settings.update_max_idx(tmp_max_idx)
    settings.update_min_idx(tmp_min_idx)

    data = np.zeros((v[0], v[1], v[2]))
    for i in range(v[0]):
        for j in range(v[1]):
            for k in range(v[2]):
                pos = np.array([((i - center_idx[0]) % v[0]) / v[0] * lattice_params[0], ((j - center_idx[1]) % v[1]) / v[1] * lattice_params[1], ((k - center_idx[2]) % v[2]) / v[2] * lattice_params[2]])
                for l in range(3):
                    if pos[l] > lattice_params[l] / 2:
                        pos[l] -= lattice_params[l]
                r = np.linalg.norm(pos)
                if r > r_max or r < r_min:
                    data[i, j, k] = 0
                else:
                    data[i, j, k] = tmp_data[i, j, k]

    print(f"load_data DEBUG: data sample (first 10 of 0,0,:): {data[0,0,:10]._value if hasattr(data[0,0,:10], '_value') else data[0,0,:10]}")
    print(f"load_data DEBUG: max of data: {np.max(data)}")
    print(f"load_data DEBUG: min of data: {np.min(data)}")
    os.makedirs("cache", exist_ok=True)
    np.save("cache/data_modified.npy", data, allow_pickle=True)
    log = {
        "center_idx": center_idx,
        "lattice_params": lattice_params,
        "v": v,
        "max_idx": settings.max_idx,
        "min_idx": settings.min_idx,
        "r_min": r_min,
        "r_max": r_max,
    }
    # with open("cache/log.json", "w") as f:
    #     json.dump(log, f)

    return pick_partial_data(data, settings)
