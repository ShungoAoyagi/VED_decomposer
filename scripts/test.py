import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.calc_Zeff import calculate_Zeff
from src.tasks.pre_processing.settings import import_settings

# settings = import_settings("settings/settings.yaml")

Zeffs = calculate_Zeff()

print(Zeffs)