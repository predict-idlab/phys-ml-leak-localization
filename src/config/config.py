import pathlib

import src

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"
RESULTS_DIR = PACKAGE_ROOT / "results"
