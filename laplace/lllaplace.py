from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from laplace.baselaplace import DiagLaplace, FullLaplace, KronLaplace, ParametricLaplace
from laplace.curvature.curvature import CurvatureInterface
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.likelihood import Likelihood as LikelihoodModule
from laplace.utils import FeatureExtractor, Kron
from laplace.utils.feature_extractor import FeatureReduction

__all__ = [
    "LLLaplace",
    "FullLLLaplace",
    "KronLLLaplace",
    "DiagLLLaplace",
]


class LLLaplace(ParametricLaplace):
    """Last-layer Laplace approximations (parametric) for neural nets."""

    _key = ("last_layer", None)

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        feature_reduction: FeatureReduction | str | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = CurvlinopsGGN,
        last_layer_name: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ):
        self.H = None
        super().__init__(
            model,
            likelihood,
            sigma_noise=sigma_noise,
            prior_precision=1.0,
            prior_mean=0.0,
            temperature=temperature,
            enable_backprop=enable_backprop,
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        self.model = FeatureExtractor(
            deepcopy(model),
            last_layer_name=last_layer_name,
            enable_backprop=enable_backprop,
            feature_reduction=feature_reduction,
        )

        if self.model.last_layer is None:
            self.mean: torch.Tensor | None = None
            self.n_params: int | None = None
            self.n_layers: int | None = None
            self._prior_precision: float | torch.Tensor = prior_precision
            self._prior_mean: float | torch.Tensor = prior_mean
        else:
            self.n_params = len(parameters_to_vector(self.model.last_layer.parameters()))
            self.n_layers = len(list(self.model.last_layer.parameters()))
            self.prior_precision = prior_precision
            self.prior_mean = prior_mean
            self.mean: float | torch.Tensor = self.prior_mean
            self._init_H()

        self._backend_kwargs["last_layer"] = True
        self._last_layer_name: str | None = last_layer_name

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        if not override:
            raise ValueError("Last-layer Laplace does not support `override=False`.")

        self.model.eval()

        if self.model.last_layer is None:
            self.data: tuple[torch.Tensor, torch.Tensor] | MutableMapping = next(
                iter(train_loader)
            )
            self._find_last_layer(self.data)
            params: torch.Tensor = parameters_to_vector(
                self.model.last_layer.parameters()
            ).detach()
            self.n_params = len(params)
            self.n_layers = len(list(self.model.last_layer.parameters()))
            self.prior_precision = self._prior_precision
            self.prior_mean = self._prior_mean
            self._init_H()

        super().fit(train_loader, override=override, progress_bar=progress_bar)
        self.mean = parameters_to_vector(self.model.last_layer.parameters())

        if not self.enable_backprop:
            self.mean = self.mean.detach()

    def _glm_predictive_distribution(
        self,
        X: torch.Tensor | MutableMapping,
        joint: bool = False,
        diagonal_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if joint:
            Js, f_mu = self.backend.last_layer_jacobians(X, self.enable_backprop)
            f_mu = f_mu.flatten()
            f_var = self.functional_covariance(Js)
        elif diagonal_output:
            try:
                f_mu, f_var = self.functional_variance_fast(X)
            except NotImplementedError:
                Js, f_mu = self.backend.last_layer_jacobians(X, self.enable_backprop)
                f_var = self.functional_variance(Js).diagonal(dim1=-2, dim2=-1)
        else:
            Js, f_mu = self.backend.last_layer_jacobians(X, self.enable_backprop)
            f_var = self.functional_variance(Js)

        return (
            (f_mu.detach(), f_var.detach())
            if not self.enable_backprop
            else (f_mu, f_var)
        )

    def _nn_predictive_samples(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        fs = []
        feats = None
        for sample in self.sample(n_samples, generator):
            vector_to_parameters(sample, self.model.last_layer.parameters())
            if feats is None:
                f, feats = self.model.forward_with_features(
                    X.to(self._device), **model_kwargs
                )
            else:
                f = self.model.last_layer(feats)
            fs.append(f.detach() if not self.enable_backprop else f)

        vector_to_parameters(self.mean, self.model.last_layer.parameters())
        fs = torch.stack(fs)
        return self._likelihood.transform_function_samples(fs)

    @property
    def prior_precision_diag(self) -> torch.Tensor:  # type: ignore[override]
        if (
            isinstance(self.prior_precision, float) or len(self.prior_precision) == 1
        ):  # scalar
            return self.prior_precision * torch.ones_like(self.mean)
        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision
        else:
            raise ValueError("Mismatch of prior and model. Diagonal or scalar prior.")

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["data"] = getattr(self, "data", None)
        state_dict["_last_layer_name"] = self._last_layer_name
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._last_layer_name != state_dict["_last_layer_name"]:
            raise ValueError("Different `last_layer_name` detected!")

        self.data = state_dict["data"]
        if self.data is not None:
            self._find_last_layer(self.data)

        super().load_state_dict(state_dict)

        params = parameters_to_vector(self.model.last_layer.parameters()).detach()
        self.n_params = len(params)
        self.n_layers = len(list(self.model.last_layer.parameters()))

    @torch.no_grad()
    def _find_last_layer(
        self, data: torch.Tensor | MutableMapping[str, torch.Tensor | Any]
    ) -> None:
        if isinstance(data, MutableMapping):
            self.model.find_last_layer(data)
        else:
            X = data[0]
            try:
                self.model.find_last_layer(X[:1].to(self._device))
            except (TypeError, AttributeError):
                self.model.find_last_layer(X.to(self._device))


class FullLLLaplace(LLLaplace, FullLaplace):
    _key = ("last_layer", "full")


class KronLLLaplace(LLLaplace, KronLaplace):
    _key = ("last_layer", "kron")

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        feature_reduction: FeatureReduction | str | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = CurvlinopsGGN,
        last_layer_name: str | None = None,
        damping: bool = False,
        backend_kwargs: dict[str, Any] | None = None,
    ):
        self.damping = damping
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            feature_reduction,
            dict_key_x,
            dict_key_y,
            backend,
            last_layer_name,
            backend_kwargs,
        )

    def _init_H(self) -> None:
        self.H = Kron.init_from_model(self.model.last_layer, self._device, self._dtype)

    def functional_variance_fast(self, X):
        raise NotImplementedError


class DiagLLLaplace(LLLaplace, DiagLaplace):
    _key = ("last_layer", "diag")

    def functional_variance_fast(self, X):
        f_mu, phi = self.model.forward_with_features(X)
        k = f_mu.shape[-1]
        b, d = phi.shape

        f_var = torch.einsum(
            "bd,kd,bd->bk", phi, self.posterior_variance[: d * k].reshape(k, d), phi
        )

        if self.model.last_layer.bias is not None:
            f_var += self.posterior_variance[-k:].reshape(1, k)

        return f_mu, f_var
