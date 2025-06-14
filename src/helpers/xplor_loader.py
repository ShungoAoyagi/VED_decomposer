import numpy as np
import os

class XplorFile:
    exists: bool = False
    path: str = ""
    v: np.ndarray[int] | None = None
    v_min: np.ndarray[int] | None = None
    v_max: np.ndarray[int] | None = None
    lattice_params: np.ndarray[float] | None = None
    data: np.ndarray | None = None  # 型は動的に決まる
    def __init__(self, path: str):
        self.path = path
        self.exists = os.path.exists(path)
        if self.exists:
            self.load()
        return
    
    def load(self):
        with open(self.path, "r") as f:
            for _ in range(3):
                next(f)
            tmp = f.readline().split()
            if len(tmp) != 9:
                raise ValueError(f"Invalid xplor file. The file {self.path} has an invalid number of columns.")
            self.v = np.zeros(3, dtype=int)
            self.v_min = np.zeros(3, dtype=int)
            self.v_max = np.zeros(3, dtype=int)
            self.v[0] = int(tmp[0])
            self.v[1] = int(tmp[3])
            self.v[2] = int(tmp[6])
            self.v_min[0] = int(tmp[1])
            self.v_min[1] = int(tmp[4])
            self.v_min[2] = int(tmp[7])
            self.v_max[0] = int(tmp[2])
            self.v_max[1] = int(tmp[5])
            self.v_max[2] = int(tmp[8])
            # 一時的にcomplex配列として読み込み
            temp_data = np.zeros((self.v[0], self.v[1], self.v[2]), dtype=complex)

            self.lattice_params = np.zeros(6, dtype=float)
            tmp = f.readline().split()
            self.lattice_params[0] = float(tmp[0])
            self.lattice_params[1] = float(tmp[1])
            self.lattice_params[2] = float(tmp[2])
            self.lattice_params[3] = float(tmp[3])
            self.lattice_params[4] = float(tmp[4])
            self.lattice_params[5] = float(tmp[5])

            for _ in range(1):
                next(f)

            for i in range(self.v_min[2], self.v[2]):
                count = 0
                tmp = f.readline().split()
                for j in range(self.v_min[1], self.v[1]):
                    for k in range(self.v_min[0], self.v[0]):
                        if (count % 5 == 0):
                            tmp = f.readline().split()
                        try:
                            # 複素数または実数として解析
                            value_str = tmp[count % 5]
                            if 'j' in value_str or 'i' in value_str:
                                # 複素数として解析（古いiフォーマットにも対応）
                                temp_data[k, j, i] = complex(value_str.replace('i', 'j'))
                            else:
                                # 実数として解析
                                temp_data[k, j, i] = complex(float(value_str))
                        except:
                            print(tmp)
                            print(count)
                            print(k, j, i)
                            print(self.path)
                            raise Exception("Error")
                        count += 1
            
            # データが実数のみかどうかをチェック
            if np.allclose(temp_data.imag, 0.0):
                # 全ての要素の虚数部がゼロ（数値誤差範囲内）の場合、float配列に変換
                self.data = temp_data.real.astype(np.float64)
            else:
                # 複素数が含まれる場合はそのまま
                self.data = temp_data

def load_xplor(path: str) -> XplorFile:
    """
    Load xplor file
    """
    return XplorFile(path)