import logging
logging.basicConfig(level=logging.INFO)

import torch
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from bonner.computation.regression import Regression

class SklearnRegression(Regression):
    def __init__(
        self,
        type: str,
        **kwargs,
    ) -> None:
        self.type = type
        self.kwargs = kwargs
        self.model_fn = self._model_fn()
        self.model = None
        
    def _model_fn(self) -> callable:
        match self.type:
            case "sklearn_linear":
                return LinearRegression
            case "sklearn_ridge":
                return Ridge
            case "sklearn_lasso":
                return Lasso
            case "sklearn_elasticnet":
                return ElasticNet
            case "sklearn_svr":
                return SVR
            case "sklearn_tree":
                return DecisionTreeRegressor
            case "sklearn_gp":
                return GaussianProcessRegressor
            case "sklearn_rf":
                return RandomForestRegressor
            case _:
                raise ValueError(f"Invalid type: {self.type}")
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        X_np = X.numpy()
        y_np = y.numpy()
        self.model = self.model_fn(**self.kwargs)
        self.model.fit(X_np, y_np)
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.numpy()
        return torch.from_numpy(self.model.predict(X_np))
    
    def weights(self) -> torch.Tensor:
        coef = torch.from_numpy(self.model.coef_)
        if coef.dim() == 1:
            return coef
        else:
            return coef.T
        
    def to(self, device: str):
        return None