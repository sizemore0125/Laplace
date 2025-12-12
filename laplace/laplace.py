from __future__ import annotations

import torch

from laplace.baselaplace import BaseLaplace
from laplace.utils.enums import (
    HessianStructure,
    Likelihood,
    SubsetOfWeights,
)


def Laplace(
    model: torch.nn.Module,
    likelihood: Likelihood | str,
    subset_of_weights: SubsetOfWeights | str = SubsetOfWeights.ALL,
    hessian_structure: HessianStructure | str = HessianStructure.FULL,
    *args,
    **kwargs,
) -> BaseLaplace:
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression'}
    subset_of_weights : SubsetofWeights or {'subnetwork', 'all'}, default=SubsetOfWeights.ALL
        subset of weights to consider for inference
    hessian_structure : HessianStructure or str in {'diag', 'full', 'lowrank', 'gp'}, default=HessianStructure.FULL
        structure of the Hessian approximation (note that in case of 'gp',
        we are not actually doing any Hessian approximation, the inference is instead done in the functional space)
    Returns
    -------
    laplace : BaseLaplace
        chosen subclass of BaseLaplace instantiated with additional arguments
    """
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )
    laplace_map = {
        subclass._key: subclass
        for subclass in _all_subclasses(BaseLaplace)
        if hasattr(subclass, "_key")
        and subclass._key[0] != "last_layer"
    }
    if (subset_of_weights, hessian_structure) not in laplace_map:
        raise ValueError(
            f"Unsupported combination subset_of_weights={subset_of_weights} "
            f"hessian_structure={hessian_structure}"
        )
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )
