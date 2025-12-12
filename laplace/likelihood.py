from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Likelihood(torch.nn.Module, ABC):
    """Base class for Laplace likelihoods.

    Subclasses behave like torch loss modules (`forward` returns a loss)
    but must also implement prediction, sigma_noise validation, and an
    optional log_prob.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "mean",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute loss; must accept reduction in {'none','mean','sum'}."""

    def loss(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "mean",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Alias for forward to align with torch loss-style usage."""
        return self.forward(f, y, reduction=reduction, temperature=temperature)

    @abstractmethod
    def predict(
        self,
        f: torch.Tensor,
        *,
        temperature: float = 1.0,
        link: str = "identity",
    ) -> torch.Tensor:
        """Map raw model outputs to predictive quantities (probabilities or means)."""

    @abstractmethod
    def check_sigma_noise(self, sigma_noise: torch.Tensor | float) -> None:
        """Validate sigma_noise for this likelihood, raising if incompatible."""

    def log_prob(
        self, f: torch.Tensor, y: torch.Tensor, *, temperature: float = 1.0
    ) -> torch.Tensor:
        """Optional convenience; defaults to -loss when not overridden."""
        return -self.loss(f, y, reduction="none", temperature=temperature)

    def loss_factor(self) -> float:
        """Scaling factor between stored loss and log-likelihood terms."""
        return 1.0

    def log_likelihood_from_loss(
        self,
        loss: torch.Tensor,
        n_data: int,
        n_outputs: int,
        sigma_noise: torch.Tensor | float,
        H_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map stored loss to log-likelihood term."""
        return -H_factor * self.loss_factor() * loss

    def default_metric(
        self, num_outputs: int, device: torch.device
    ) -> torch.nn.Module | None:
        """Optional default torchmetrics-style metric for validation/gridsearch."""
        return None

    def is_regression_like(self) -> bool:
        """Whether this likelihood behaves like regression for variance/joint handling."""
        return False

    def is_classification_like(self) -> bool:
        return False

    def validate_output_targets(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Hook to validate shapes/dtypes of model output vs. target."""

    def transform_function_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Transform function samples to predictive space (e.g., softmax for classification)."""
        return samples

    @property
    def supports_sigma_noise_change(self) -> bool:
        """Whether sigma_noise can be changed after initialization."""
        return False

    def prediction_kind(self, fitting: bool) -> str:
        """Return the prediction mode this likelihood wants (e.g., classification/regression)."""
        return self.name

    def scatter_mean(
        self,
        Js: torch.Tensor,
        f: torch.Tensor,
        y: torch.Tensor,
        prior_mean: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:
        """Mean term for scatter; subclasses override."""
        raise NotImplementedError

    def functional_lambdas(self, f_batch: torch.Tensor) -> torch.Tensor:
        """Return diagonal Hessian blocks for functional Laplace."""
        raise NotImplementedError


class ClassificationLikelihood(Likelihood):
    """Multiclass classification via cross entropy."""

    def __init__(self) -> None:
        super().__init__(name="classification")

    def forward(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "mean",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            f / temperature, y, reduction=reduction
        )

    def predict(
        self,
        f: torch.Tensor,
        *,
        temperature: float = 1.0,
        link: str = "softmax",
    ) -> torch.Tensor:
        if link == "softmax":
            return torch.nn.functional.softmax(f / temperature, dim=-1)
        if link == "logits":
            return f / temperature
        raise ValueError(f"Unknown link '{link}' for classification likelihood.")

    def check_sigma_noise(self, sigma_noise: torch.Tensor | float) -> None:
        sigma = torch.as_tensor(sigma_noise)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if not torch.allclose(sigma, torch.ones_like(sigma)):
            raise ValueError("sigma_noise must be 1 for classification likelihood.")

    def transform_function_samples(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(samples, dim=-1)

    def default_metric(
        self, num_outputs: int, device: torch.device
    ) -> torch.nn.Module | None:
        try:
            from laplace.utils.metrics import RunningNLLMetric  # local import to avoid torch dep outside
        except Exception:
            return None
        return RunningNLLMetric().to(device)

    def is_classification_like(self) -> bool:  # type: ignore[override]
        return True

    def scatter_mean(
        self,
        Js: torch.Tensor,
        f: torch.Tensor,
        y: torch.Tensor,
        prior_mean: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        return -torch.einsum("bcp,p->bc", Js, prior_mean - mean)

    def loss_factor(self) -> float:  # type: ignore[override]
        return 1.0

    def functional_lambdas(self, f_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ps = torch.nn.functional.softmax(f_batch, dim=-1)
        return torch.diag_embed(ps) - torch.einsum("mk,mc->mck", ps, ps)

    def functional_lambdas(self, f_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ps = torch.nn.functional.softmax(f_batch, dim=-1)
        return torch.diag_embed(ps) - torch.einsum("mk,mc->mck", ps, ps)


class RegressionLikelihood(Likelihood):
    """Homoskedastic regression via mean-squared error."""

    def __init__(self) -> None:
        super().__init__(name="regression")

    def forward(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "mean",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Temperature scales variance: loss increases with 1 / temperature^2.
        return torch.nn.functional.mse_loss(f, y, reduction=reduction) / (
            temperature**2
        )

    def predict(
        self,
        f: torch.Tensor,
        *,
        temperature: float = 1.0,
        link: str = "identity",
    ) -> torch.Tensor:
        if link == "identity":
            return f
        raise ValueError(f"Unknown link '{link}' for regression likelihood.")

    def check_sigma_noise(self, sigma_noise: torch.Tensor | float) -> None:
        sigma = torch.as_tensor(sigma_noise)
        if torch.any(sigma <= 0):
            raise ValueError("sigma_noise must be positive for regression likelihood.")

    def supports_sigma_noise_change(self) -> bool:  # type: ignore[override]
        return True

    def default_metric(
        self, num_outputs: int, device: torch.device
    ) -> torch.nn.Module | None:
        try:
            import torchmetrics
        except Exception:
            return None
        return torchmetrics.MeanSquaredError(num_outputs=num_outputs).to(device)

    def validate_output_targets(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> None:  # type: ignore[override]
        if target.ndim != output.ndim:
            raise ValueError(
                f"The model's output has {output.ndim} dims but "
                f"the target has {target.ndim} dims."
            )

    def log_likelihood_from_loss(
        self,
        loss: torch.Tensor,
        n_data: int,
        n_outputs: int,
        sigma_noise: torch.Tensor | float,
        H_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:  # type: ignore[override]
        sigma_noise_tensor = torch.as_tensor(
            sigma_noise, device=device, dtype=dtype
        )
        normalizer = (
            n_data
            * n_outputs
            * torch.log(sigma_noise_tensor * torch.tensor(2.0 * torch.pi).sqrt())
        )
        return -H_factor * self.loss_factor() * loss - normalizer

    def scatter_mean(
        self,
        Js: torch.Tensor,
        f: torch.Tensor,
        y: torch.Tensor,
        prior_mean: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        return y - (f + torch.einsum("bcp,p->bc", Js, prior_mean - mean))

    def loss_factor(self) -> float:  # type: ignore[override]
        return 0.5

    def is_regression_like(self) -> bool:  # type: ignore[override]
        return True

    def functional_lambdas(self, f_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, C = f_batch.shape
        eye = torch.eye(C, device=f_batch.device, dtype=f_batch.dtype)
        return torch.unsqueeze(eye, 0).repeat(b, 1, 1)


class RewardModelingLikelihood(Likelihood):
    """Bradley-Terry style preference learning (classification loss, placeholder predict)."""

    def __init__(self) -> None:
        super().__init__(name="reward_modeling")

    def forward(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "mean",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Expect f shape (batch, 2) with y in {0,1}; standard CE on pairwise scores.
        return torch.nn.functional.cross_entropy(
            f / temperature, y, reduction=reduction
        )

    def predict(
        self,
        f: torch.Tensor,
        *,
        temperature: float = 1.0,
        link: str = "softmax",
    ) -> torch.Tensor:
        return torch.nn.functional.softmax(f / temperature, dim=-1)

    def check_sigma_noise(self, sigma_noise: torch.Tensor | float) -> None:
        sigma = torch.as_tensor(sigma_noise)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if not torch.allclose(sigma, torch.ones_like(sigma)):
            raise ValueError("sigma_noise must be 1 for reward modeling likelihood.")

    def transform_function_samples(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(samples, dim=-1)

    def default_metric(
        self, num_outputs: int, device: torch.device
    ) -> torch.nn.Module | None:
        try:
            from laplace.utils.metrics import RunningNLLMetric  # local import
        except Exception:
            return None
        return RunningNLLMetric().to(device)

    def prediction_kind(self, fitting: bool) -> str:  # type: ignore[override]
        return "classification" if fitting else "regression"

    def scatter_mean(
        self,
        Js: torch.Tensor,
        f: torch.Tensor,
        y: torch.Tensor,
        prior_mean: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        return -torch.einsum("bcp,p->bc", Js, prior_mean - mean)

    def is_classification_like(self) -> bool:  # type: ignore[override]
        return True
