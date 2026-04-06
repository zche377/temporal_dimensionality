import logging
logging.basicConfig(level=logging.INFO)

import torch
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

from lib.computation.classifiers import Classifier



class SklearnClassifier(Classifier):
    def __init__(
        self,
        type: str,
        **kwargs,
    ) -> None:
        self.type = type
        self.kwargs = kwargs
        self.model_fn = self._model_fn()
        self.models = None
        
    def _model_fn(self) -> callable:
        match self.type:
            case "sklearn_lda":
                return LinearDiscriminantAnalysis
            case "sklearn_qda":
                return QuadraticDiscriminantAnalysis
            case "sklearn_logistic":
                return LogisticRegression
            case "sklearn_svm":
                return SVC
            case "sklearn_tree":
                return DecisionTreeClassifier
            case "sklearn_nb":
                return GaussianNB
            case "sklearn_gp":
                return GaussianProcessClassifier
            case _:
                raise ValueError(f"Invalid type: {self.type}")
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if y.dim() == 1:
            self.models = [self.model_fn(**self.kwargs)]
            self.models[0].fit(X, y)
        else:
            self.models = []
            for i in range(y.shape[1]):
                model = self.model_fn(**self.kwargs)
                model.fit(X, y[:, i])
                self.models.append(model)
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            prediction = torch.from_numpy(model.predict(X))
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(-1)
            predictions.append(prediction)
        return torch.cat(predictions, dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            prediction = torch.from_numpy(model.predict_proba(X))
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(-1)
            predictions.append(prediction)
        return torch.cat(predictions, dim=1)
    
    def weights(self) -> torch.Tensor:
        coefs = [
            torch.from_numpy(m.ceof_).T
            for m in self.models
        ]
        return torch.cat(coefs, dim=-1)
    
    def to(self, device: str):
        return None