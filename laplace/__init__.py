"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
.. include:: ../examples/huggingface_example.md
"""

from laplace.baselaplace import (
    BaseLaplace,
    DiagLaplace,
    FullLaplace,
    ParametricLaplace,
)
from laplace.laplace import Laplace
from laplace.subnetlaplace import DiagSubnetLaplace, FullSubnetLaplace, SubnetLaplace
from laplace.utils.enums import (
    HessianStructure,
    Likelihood,
    LinkApprox,
    PredType,
    PriorStructure,
    SubsetOfWeights,
    TuningMethod,
)

__all__ = [
    "Laplace",  # direct access to all Laplace classes via unified interface
    "BaseLaplace",
    "ParametricLaplace",  # base-class and its (first-level) subclasses
    "FullLaplace",
    "DiagLaplace",
    "SubnetLaplace",  # base-class subnetwork
    "FullSubnetLaplace",
    "DiagSubnetLaplace",  # subnetwork
    # Enums
    "SubsetOfWeights",
    "HessianStructure",
    "Likelihood",
    "PredType",
    "LinkApprox",
    "TuningMethod",
    "PriorStructure",
]
