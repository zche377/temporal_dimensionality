import logging
logging.basicConfig(level=logging.INFO)

import torch
import numpy as np
from lib.computation.metrics import compute_metric
from torchmetrics.functional import pearson_corrcoef

from bonner.computation.regression import Regression

class RidgeRegression(Regression):
    def __init__(
        self,
        alpha: (float | list[float] | str) = "loocv",
        fit_intercept: bool = True,
        scale: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alphas = [alpha]
        elif isinstance(alpha, list):
            self.alphas = alpha
        elif alpha == "loocv":
            self.alphas = np.logspace(-4, 4, 10).tolist()
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.coef_ = None
        self.intercept = None
        self.scale_std = None

    def to(self, device: str = None) -> None:
        if device is not None:
            self.device = device
        if self.coef_ is not None:
            self.coef_ = self.coef_.to(self.device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(self.device)
        if self.scale_std is not None:
            self.scale_std = self.scale_std.to(self.device)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = torch.clone(X).to(self.device)
        y = torch.clone(y).to(self.device)

        if self.fit_intercept:
            X_mean = X.mean(dim=0)
            y_mean = y.mean(dim=0)
            X -= X_mean
            y -= y_mean
        else:
            X_mean = torch.zeros(X.shape[1], device=self.device)
            y_mean = torch.zeros(1, device=self.device)

        if self.scale:
            self.scale_std = X.std(dim=0)
            X /= self.scale_std
        else:
            self.scale_std = torch.ones(X.shape[1], device=self.device)

        sqrt_sw = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
        eigvals, Q, QT_y = self._eigen_decompose_gram(X, y, sqrt_sw)

        n_targets = y.shape[1] if y.ndim > 1 else 1
        best_alpha, best_coef, best_score = torch.zeros(n_targets), torch.zeros_like(QT_y), torch.full((n_targets,), float('-inf'))
        best_y_pred = None

        for alpha in self.alphas:
            G_inverse_diag, coef = self._solve_eigen_gram(alpha, y, sqrt_sw, eigvals, Q, QT_y)
            y_pred = y - (coef / G_inverse_diag)

            score = compute_metric(metric="pearsonr", y_true=[y], y_predicted=[y_pred], score_across_folds=True)
            if n_targets > 1:
                update_mask = score > best_score
                best_alpha[update_mask] = alpha
                best_coef[:, update_mask] = coef[:, update_mask]
                best_score[update_mask] = score[update_mask]
                if best_y_pred is None:
                    best_y_pred = y_pred.clone()
                else:
                    best_y_pred[:, update_mask] = y_pred[:, update_mask]
            else:
                if score > best_score:
                    best_alpha, best_coef, best_score, best_y_pred = alpha, coef, score, y_pred

        self.alpha = best_alpha
        self.dual_coef_ = best_coef
        self.coef_ = self.dual_coef_.t().matmul(X)
        self.cv_y_pred_ = best_y_pred

        if self.fit_intercept:
            self.coef_ = self.coef_ / self.scale_std
            self.intercept = y_mean - torch.matmul(X_mean, self.coef_.t())
        else:
            self.intercept = torch.zeros(1, device=self.coef_.device)


    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.clone(X).to(self.device)
        X /= self.scale_std
        return X.matmul(self.coef_.t()) + self.intercept

    def weights(self) -> torch.Tensor:
        return self.coef_.t()
    
    @staticmethod
    def _decomp_diag(v_prime, Q):
        return (v_prime * Q**2).sum(axis=-1)

    @staticmethod
    def _diag_dot(D, B):
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B

    @staticmethod
    def _find_smallest_angle(query, vectors):
        abs_cosine = torch.abs(torch.matmul(query, vectors))
        return torch.argmax(abs_cosine).item()

    def _compute_gram(self, x, sqrt_sw):
        return x.matmul(x.T)

    def _eigen_decompose_gram(self, x, y, sqrt_sw):
        K = self._compute_gram(x, sqrt_sw)
        if self.fit_intercept:
            K += torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)
        QT_y = torch.matmul(Q.T, y)
        return eigvals, Q, QT_y

    def _solve_eigen_gram(self, alpha, y, sqrt_sw, eigvals, Q, QT_y):
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / torch.linalg.norm(sqrt_sw)
            intercept_dim = self._find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0  # Cancel regularization for the intercept

        c = torch.matmul(Q, self._diag_dot(w, QT_y))
        G_inverse_diag = self._decomp_diag(w, Q)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c