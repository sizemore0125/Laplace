from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from curvlinops import (
    EFLinearOperator,
    FisherMCLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
)
from curvlinops._base import _LinearOperator
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.likelihood import Likelihood as LikelihoodModule


class CurvlinopsInterface(CurvatureInterface):
    """Interface for Curvlinops backend. <https://github.com/f-dangel/curvlinops>"""

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ) -> None:
        super().__init__(
            model, likelihood, subnetwork_indices, dict_key_x, dict_key_y
        )
        self._likelihood = likelihood

    def _loss_closure(self):
        def loss_fn(f, target):
            return self._likelihood.loss(f, target, reduction="sum")

        loss_fn.reduction = "sum"
        return loss_fn

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        raise NotImplementedError

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fallback to torch.func backend for SubnetLaplace
        if self.subnetwork_indices is not None:
            return super().full(x, y, **kwargs)

        loss_closure = self._loss_closure()
        curvlinops_kwargs = {k: v for k, v in kwargs.items() if k != "N"}
        if isinstance(x, (dict, MutableMapping)):
            curvlinops_kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        linop = self._linop_context(
            self.model,
            loss_closure,
            self.params,
            [(x, y)],
            check_deterministic=False,
            **curvlinops_kwargs,
        )

        p = next(self.model.parameters())
        H = torch.as_tensor(
            linop @ torch.eye(linop.shape[0]), device=p.device, dtype=p.dtype
        )

        f = self.model(x)
        loss = loss_closure(f, y)

        return self.factor * loss.detach(), self.factor * H


class CurvlinopsGGN(CurvlinopsInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Curvlinops."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
    ) -> None:
        super().__init__(
            model, likelihood, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return FisherMCLinearOperator if self.stochastic else GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops."""

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return EFLinearOperator


class CurvlinopsHessian(CurvlinopsInterface):
    """Implementation of the full Hessian using Curvlinops."""

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return HessianLinearOperator
