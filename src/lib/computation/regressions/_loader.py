from typing import Any

from lib.computation.regressions import (
    RidgeRegression,
    SklearnRegression
)

from bonner.computation.regression import (
    Regression,
    LinearRegression,
    PLSRegression,
    SGDLinearRegression
)



def load_regression(regression: str, **kwargs: Any) -> Regression:
    if regression.split("_")[0] == "sklearn":
        if "device" in kwargs.keys():
            kwargs.pop("device")
        return SklearnRegression(type=regression, **kwargs)
    match regression:
        case "ridge":
            return RidgeRegression(**kwargs)
        case "linear":
            return LinearRegression(**kwargs)
        case "pls":
            return PLSRegression(**kwargs)
        case "sgd_linear":
            return SGDLinearRegression(**kwargs)
        case _:
            raise ValueError(f"Invalid regression: {regression}")
        
def list_all_regressions() -> list[str]:
    return ["ridge", "sklearn_linear", "sklearn_ridge", "sklearn_lasso", "sklearn_elasticnet", "sklearn_svr", "sklearn_tree", "sklearn_gp", "sklearn_rf", "linear", "pls", "sgd_linear"]
