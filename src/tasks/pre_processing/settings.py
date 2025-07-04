from prefect import task
import yaml
from src.utils import CustomError, ErrorHandler, ErrorLevel, ErrorCode
import numpy as np

orbital_magnetic_number = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
}

class Settings:
    def __init__(self, setting_path: str):
        self.setting_path = setting_path
        self.error_handler = ErrorHandler()
        self.import_settings()

    atom_name: str = "Fe"
    orbital_set: list[str] = ["3d", "4s", "4p"]
    orb_idx_set: list[tuple[int, int]] = []
    spin: int = 0
    center: list[float] = [0, 0, 0]
    r_min: float = 0.0 # Å unit
    r_max: float = 1.0 # Å unit
    basis_set: list[list[float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    d_min: float = 0.28 # Å unit
    v: list[int] = [100, 100, 100] # mesh size of data
    lattice_params: list[float] = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0] # lattice parameters
    center_idx: list[int] = [0, 0, 0] # index of center of data
    max_idx: list[int] = [0, 0, 0] # max index of data
    min_idx: list[int] = [0, 0, 0] # min index of data
    fit_orbital: str = "3d"

    def convert_orbital_set(self) -> list[tuple[int, int]]:
        for orb in self.orbital_set:
            orb_idx = int(orb[0])
            orb_mag = orbital_magnetic_number[orb[1]]
            self.orb_idx_set.append((orb_idx, orb_mag))

    def import_settings(self) -> CustomError | None:
        try:
            with open(self.setting_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            if settings is None:
                settings = {}

        except FileNotFoundError:
            self.error_handler.handle(
                f"Settings file '{self.setting_path}' not found. Using default values.",
                ErrorCode.NOT_FOUND,
                ErrorLevel.ERROR,
                {"file_path": self.setting_path}
            )
            settings = {}
        except yaml.YAMLError:
            self.error_handler.handle(
                f"Invalid YAML format in settings file '{self.setting_path}'. Using default values.",
                ErrorCode.VALIDATION,
                ErrorLevel.ERROR,
                {"file_path": self.setting_path}
            )
            settings = {}

        if "atom_name" not in settings:
            self.error_handler.handle(
                "There is no atom_name in settings. Therefore, the default value (Fe) is used.",
                ErrorCode.NOT_FOUND,
                ErrorLevel.WARNING,
                {"available_fields": list(settings.keys())}
            )
        else:
            self.atom_name = settings["atom_name"]

        if "orbital_set" not in settings:
            self.error_handler.handle(
                "There is no orbital_set in settings. Therefore, the default value (3d, 4s, 4p) is used.",
                ErrorCode.NOT_FOUND,
                ErrorLevel.WARNING,
                {"available_fields": list(settings.keys())}
            )
        else:
            self.orbital_set = settings["orbital_set"]
        self.convert_orbital_set()

        if "spin" in settings:
            self.spin = settings["spin"]
        
        if "center" in settings:
            self.center = settings["center"]
            if len(self.center) != 3:
                self.error_handler.handle(
                    "The center must be a 3D coordinate. Using default value [0, 0, 0].",
                    ErrorCode.VALIDATION,
                    ErrorLevel.WARNING,
                    {"center": self.center}
                )
                self.center = [0, 0, 0]
        
        if "r_min" in settings:
            self.r_min = settings["r_min"]

        if "r_max" in settings:
            self.r_max = settings["r_max"]

        if "fit_orbital" in settings:
            self.fit_orbital = settings["fit_orbital"]

        if "basis_set" in settings:
            self.basis_set = settings["basis_set"]
            if len(self.basis_set) != 3:
                self.error_handler.handle(
                    "The basis_set must be a list of 3 lists. The default value is used.",
                    ErrorCode.VALIDATION,
                    ErrorLevel.WARNING,
                    {"available_fields": list(settings.keys())}
                )
            else:
                flag = False
                for i in range(3):
                    if len(self.basis_set[i]) != 3:
                        self.error_handler.handle(
                            "The basis_set must be a list of 3 lists. The default value is used.",
                            ErrorCode.VALIDATION,
                            ErrorLevel.WARNING,
                            {"available_fields": list(settings.keys())}
                        )
                        flag = True
                        break
                if flag:
                    self.basis_set = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                else:
                    for i in range(3):
                        for j in range(3):
                            self.basis_set[i][j] = float(self.basis_set[i][j])

                    for i in range(3):
                        norm = np.linalg.norm(self.basis_set[i])
                        for j in range(3):
                            self.basis_set[i][j] = self.basis_set[i][j] / norm
    
    def update_v(self, new_v: list[int]) -> None:
        self.v = new_v

    def update_lattice_params(self, new_lattice_params: list[float]) -> None:
        self.lattice_params = new_lattice_params

    def update_center_idx(self, new_center_idx: list[int]) -> None:
        self.center_idx = new_center_idx

    def update_max_idx(self, new_max_idx: list[int]) -> None:
        self.max_idx = new_max_idx
    
    def update_min_idx(self, new_min_idx: list[int]) -> None:
        self.min_idx = new_min_idx

@task(name="import settings")
def import_settings(setting_path: str) -> Settings:
    """
    load settings from yaml file
    """
    return Settings(setting_path)
