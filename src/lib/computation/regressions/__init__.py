__all__ = [
    "RidgeRegression",
    "SklearnRegression",
    "load_regression",
    "list_all_regressions"
]

from lib.computation.regressions._ridge_regression import RidgeRegression
from lib.computation.regressions._sklearn_regressions import SklearnRegression
from lib.computation.regressions._loader import (
    load_regression,
    list_all_regressions
)