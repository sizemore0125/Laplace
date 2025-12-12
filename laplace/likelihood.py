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
