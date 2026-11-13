import logging
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np
from lib.computation.classifiers import Classifier
from lib.computation.metrics import compute_metric
from lib.utilities import SEED
from bonner.computation.regression import create_stratified_splits



class LDAClassifier(Classifier):
    def __init__(
        self,
        shrinkage: (float | str),
        fit_intercept: bool = True,
        scale: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self.fit_intercept = fit_intercept
        self.shrinkage = shrinkage
        self.scale = scale
        self.device = device
        self.means = None
        self.covariance = None
        self.priors = None
        self.coefficients = None
        self.intercept = None
        self.scale_std = None

    def to(self, device: str = None) -> None:
        if device is not None:
            self.device = device
        self.coefficients = [coef.to(self.device) for coef in self.coefficients]
        self.intercepts = [intercept.to(self.device) for intercept in self.intercepts]
        self.scale_std = self.scale_std.to(self.device)
    
    def oas(self, X: torch.Tensor) -> float:
        X = X - X.mean(dim=0)
        if len(X.shape) == 2 and X.shape[1] == 1:
            return (X**2).mean(), 0.0

        n_samples, n_features = X.shape
        X = X - X.mean(dim=0)
        emp_cov = torch.cov(X.T)
        alpha = torch.mean(emp_cov**2)
        mu = torch.trace(emp_cov) / n_features
        mu_squared = mu**2
        num = alpha + mu_squared
        den = (n_samples + 1) * (alpha - mu_squared / n_features)
        shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
        return shrinkage
    
    def cv_shrinkage(self, x: torch.Tensor, y: torch.Tensor) -> float:
        shrinkages = np.logspace(-4, -.1, 20)
        splits = create_stratified_splits(y=y.cpu().numpy(), n_folds=3, shuffle=True, seed=SEED)
        y = y.unsqueeze(-1)
        
        best_shrinkage = None
        best_score = float('-inf')
        
        for shrinkage in shrinkages:
            y_true, y_predicted = [], []
            for indices_test in splits:
                indices_train = np.setdiff1d(np.arange(y.shape[-2]), np.array(indices_test))
                x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
                y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]
                
                model = LDAClassifier(fit_intercept=self.fit_intercept, shrinkage=shrinkage, scale=self.scale, device=self.device)
                model.fit(x_train, y_train)
                y_true.append(y_test)
                y_predicted.append(model.predict(x_test))
                
            score = compute_metric(metric="accuracy", y_true=y_true, y_predicted=y_predicted, score_across_folds=True)
            if score > best_score:
                best_score = score
                best_shrinkage = shrinkage
        
        return best_shrinkage
        
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        X = torch.clone(X).to(self.device)
        y = torch.clone(y).to(self.device)
        n_features = X.shape[1]
        n_columns = y.shape[1]
        
        if self.scale:
            self.scale_std = X.std(dim=0)
            X /= self.scale_std
        else:
            self.scale_std = torch.ones(n_features, device=X.device)

        self.means = []
        self.covariances = []
        self.priors = []
        self.coefficients = []
        self.intercepts = []

        for col in range(n_columns):
            y_col = y[:, col]
            classes = torch.unique(y_col)
            n_classes = len(classes)

            means = torch.zeros((n_classes, n_features), device=X.device)
            covariance = torch.zeros((n_features, n_features), device=X.device)
            priors = torch.zeros(n_classes, device=X.device)

            for idx, cls in enumerate(classes):
                X_cls = X[y_col == cls]
                means[idx] = X_cls.mean(dim=0)
                centered_X_cls = X_cls - means[idx]
                covariance += centered_X_cls.t() @ centered_X_cls
                priors[idx] = len(X_cls) / len(X)

            covariance /= len(X) - n_classes

            match self.shrinkage:
                case "loocv":
                    shrinkage = self.cv_shrinkage(X, y_col)
                case "oas":
                    shrinkage = self.oas(X)
                case _:
                    shrinkage = self.shrinkage

            if shrinkage > 0.0:
                covariance = (1 - shrinkage) * covariance + shrinkage * torch.eye(n_features, device=X.device) * torch.diag(covariance).mean()

            coefficients = torch.linalg.inv(covariance) @ means.t()
            if self.fit_intercept:
                intercept = -0.5 * torch.diag(means @ coefficients) + torch.log(priors)
            else:
                intercept = torch.zeros(n_classes, device=X.device)

            self.means.append(means)
            self.covariances.append(covariance)
            self.priors.append(priors)
            self.coefficients.append(coefficients)
            self.intercepts.append(intercept)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X /= self.scale_std
        predictions = []

        for coef, intercept in zip(self.coefficients, self.intercepts):
            decision_values = X @ coef + intercept
            predictions.append(torch.argmax(decision_values, dim=1, keepdim=True))

        return torch.cat(predictions, dim=1).to(self.device)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X /= self.scale_std
        all_probabilities = []

        for coef, intercept in zip(self.coefficients, self.intercepts):
            decision_values = X @ coef + intercept
            exp_vals = torch.exp(decision_values)
            probabilities = exp_vals / exp_vals.sum(dim=1, keepdim=True)
            all_probabilities.append(probabilities)

        return torch.cat(all_probabilities, dim=1).to(self.device)
    
    def weights(self) -> torch.Tensor:
        if self.coefficients is None:
            return self.coefficients
        else:
            return torch.cat(self.coefficients, dim=-1)
