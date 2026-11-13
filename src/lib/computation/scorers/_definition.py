"""
    Adapted code from Raj Magesh
"""
import logging

logging.basicConfig(level=logging.INFO)
from abc import ABC, abstractmethod

import xarray as xr
from bonner.computation.xarray import align_source_to_target

class TrainTestScorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        return self._score(
            predictor_train=predictor_train,
            target_train=target_train,
            predictor_test=predictor_test,
            target_test=target_test,
            target_dim=target_dim
        )

    @abstractmethod
    def _score(
        self, 
        *, 
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        pass


class TrainTestGeneralizationScorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        predictor_dim: str,
        target_dim: str,
    ) -> xr.Dataset:
        return self._score(
            predictor_train=predictor_train,
            target_train=target_train,
            predictor_test=predictor_test,
            target_test=target_test,
            target_dim=target_dim,
            predictor_dim=predictor_dim,
        )

    @abstractmethod
    def _score(
        self, 
        *, 
        predictor_train: xr.DataArray,
        target_train: xr.DataArray,
        predictor_test: xr.DataArray,
        target_test: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        pass


class Scorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictor: xr.DataArray, 
        target: xr.DataArray,
        target_dim: str,
    ) -> xr.Dataset:
        return self._score(
            predictor=predictor, 
            target=target, 
            target_dim=target_dim
        )

    @abstractmethod
    def _score(
        self, 
        *, 
        predictor: xr.DataArray, 
        target: xr.DataArray, 
        target_dim: str,
    ) -> xr.Dataset:
        pass
    
    

class GeneralizationScorer(ABC):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(
        self, 
        *, 
        predictors: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        return self._score(
            predictors=predictors, 
            target=target, 
            target_dim=target_dim,
            predictor_dim=predictor_dim,
        )

    @abstractmethod
    def _score(
        self, 
        *, 
        predictors: xr.DataArray,
        target: xr.DataArray,
        target_dim: str,
        predictor_dim: str,
    ) -> xr.Dataset:
        pass
    
