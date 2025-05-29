import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import import_settings
from src.tasks.pre_processing.load_data import load_data

settings = import_settings("data/input/settings.yaml")

data = load_data("data/input/data.xplor", settings)

# # Zeffs = calculate_Zeff()

# # print(Zeffs)
