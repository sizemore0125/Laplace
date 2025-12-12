from __future__ import annotations

from typing import Any, Callable, MutableMapping

import torch
from torch import nn
import warnings
from laplace.likelihood import Likelihood as LikelihoodModule


class CurvatureInterface:
    """Interface to access curvature for a model and corresponding likelihood.
    A `CurvatureInterface` must inherit from this baseclass and implement the
    necessary functions `jacobians`, `full`, and `diag`.
    The interface might be extended in the future to account for other curvature
    structures, for example, a block-diagonal one.

    Parameters
    ----------
    model : torch.nn.Module
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    subnetwork_indices : torch.LongTensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ):
        if not isinstance(likelihood, LikelihoodModule):
            raise ValueError("CurvatureInterface now requires a Likelihood object.")
        self.likelihood: LikelihoodModule = likelihood
        self.model: nn.Module = model
        self.subnetwork_indices: torch.LongTensor | None = subnetwork_indices
        self.dict_key_x = dict_key_x
        self.dict_key_y = dict_key_y
        self.lossfunc = self._make_loss_fn(likelihood)
        self.factor = likelihood.loss_factor()

        self.params: list[nn.Parameter] = [
            p for p in self._model.parameters() if p.requires_grad
        ]
        self.params_dict: dict[str, nn.Parameter] = {
            k: v for k, v in self._model.named_parameters() if v.requires_grad
        }
        self.buffers_dict: dict[str, torch.Tensor] = {
            k: v for k, v in self.model.named_buffers()
        }

    @property
    def _model(self) -> nn.Module:
        return self.model

    def _make_loss_fn(self, likelihood: LikelihoodModule):
        def loss_fn(f, y):
            return likelihood.loss(f, y, reduction="sum")

        # mimic torch loss modules' `reduction` attribute expected by Curvlinops/KFAC
        loss_fn.reduction = "sum"
        return loss_fn

    def jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\),
        via torch.func.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        enable_backprop : bool, default = False
            whether to enable backprop through the Js and f w.r.t. x

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """

        def model_fn_params_only(params_dict, buffers_dict):
            out = torch.func.functional_call(self.model, (params_dict, buffers_dict), x)
            return out, out

        Js, f = torch.func.jacrev(model_fn_params_only, has_aux=True)(
            self.params_dict, self.buffers_dict
        )

        # Concatenate over flattened parameters
        Js = [
            j.flatten(start_dim=-p.dim())
            for j, p in zip(Js.values(), self.params_dict.values())
        ]
        Js = torch.cat(Js, dim=-1)

        if self.subnetwork_indices is not None:
            Js = Js[:, :, self.subnetwork_indices]

        return (Js, f) if enable_backprop else (Js.detach(), f.detach())

    def gradients(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any], y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batch gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at
        current parameter \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        loss : torch.Tensor
        """

        def loss_single(x, y, params_dict, buffers_dict):
            """Compute the gradient for a single sample."""
            x, y = x.unsqueeze(0), y.unsqueeze(0)  # vmap removes the batch dimension
            output = torch.func.functional_call(
                self.model, (params_dict, buffers_dict), x
            )
            loss = torch.func.functional_call(self.lossfunc, {}, (output, y))
            return loss, loss

        grad_fn = torch.func.grad(loss_single, argnums=2, has_aux=True)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, None, None))

        batch_grad, batch_loss = batch_grad_fn(
            x, y, self.params_dict, self.buffers_dict
        )
        Gs = torch.cat([bg.flatten(start_dim=1) for bg in batch_grad.values()], dim=1)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]

        loss = batch_loss.sum(0)

        return Gs, loss

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ):
        """Compute a dense curvature (approximation) in the form of a \\(P \\times P\\) matrix
        \\(H\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            Hessian approximation `(parameters, parameters)`
        """
        raise NotImplementedError

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ):
        """Compute a diagonal Hessian approximation to \\(H\\) and is represented as a
        vector of the dimensionality of parameters \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            vector representing the diagonal of H
        """
        raise NotImplementedError

    functorch_jacobians = jacobians


class GGNInterface(CurvatureInterface):
    """Generalized Gauss-Newton or Fisher Curvature Interface.
    The GGN is equal to the Fisher information for the available likelihoods.
    In addition to `CurvatureInterface`, methods for Jacobians are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    stochastic : bool, default=False
        Fisher if stochastic else GGN
    num_samples: int, default=1
        Number of samples used to approximate the stochastic Fisher
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: LikelihoodModule,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
        num_samples: int = 1,
    ) -> None:
        self.stochastic: bool = stochastic
        self.num_samples: int = num_samples

        super().__init__(
            model, likelihood, subnetwork_indices, dict_key_x, dict_key_y
        )

    def _get_mc_functional_fisher(self, f: torch.Tensor) -> torch.Tensor:
        """Approximate the Fisher's middle matrix (expected outer product of the functional gradient)
        using MC integral with `self.num_samples` many samples.
        """
        F = 0

        for _ in range(self.num_samples):
            grad_sample = self.likelihood.sample_functional_grad(f)

            F += (
                1
                / self.num_samples
                * torch.einsum("bc,bk->bck", grad_sample, grad_sample)
            )

        return F

    def _get_functional_hessian(self, f: torch.Tensor) -> torch.Tensor | None:
        H = self.likelihood.functional_hessian(f)
        if H is None:
            warnings.warn(
                "functional_hessian returned None; falling back to identity/J^T J. "
                "Implement functional_hessian in your likelihood or use a Fisher backend "
                "for better curvature.",
                RuntimeWarning,
            )
        return H

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            GGN `(parameters, parameters)`
        """
        Js, f = self.jacobians(x)
        H_lik = (
            self._get_mc_functional_fisher(f)
            if self.stochastic
            else self._get_functional_hessian(f)
        )

        if H_lik is not None:
            H = torch.einsum("bcp,bck,bkq->pq", Js, H_lik, Js)
        else:  # The case of exact GGN for regression
            H = torch.einsum("bcp,bcq->pq", Js, Js)
        loss = self.factor * self.lossfunc(f, y)

        return loss.detach(), H.detach()

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Js, f = self.jacobians(x)
        loss = self.factor * self.lossfunc(f, y)

        H_lik = (
            self._get_mc_functional_fisher(f)
            if self.stochastic
            else self._get_functional_hessian(f)
        )

        if H_lik is not None:
            H = torch.einsum("bcp,bck,bkp->p", Js, H_lik, Js)
        else:  # The case of exact GGN for regression
            H = torch.einsum("bcp,bcp->p", Js, Js)

        return loss.detach(), H.detach()


class EFInterface(CurvatureInterface):
    """Interface for Empirical Fisher as Hessian approximation.
    In addition to `CurvatureInterface`, methods for gradients are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the full EF \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ef}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ef : torch.Tensor
            EF `(parameters, parameters)`
        """
        Gs, loss = self.gradients(x, y)
        Gs, loss = Gs.detach(), loss.detach()
        H_ef = torch.einsum("bp,bq->pq", Gs, Gs)
        return self.factor * loss.detach(), self.factor * H_ef

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gs is (batchsize, n_params)
        Gs, loss = self.gradients(x, y)
        Gs, loss = Gs.detach(), loss.detach()
        diag_ef = torch.einsum("bp,bp->p", Gs, Gs)
        return self.factor * loss, self.factor * diag_ef
