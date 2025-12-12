import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.baselaplace import DiagLaplace, FullLaplace
from laplace.laplace import Laplace

torch.manual_seed(240)
torch.set_default_dtype(torch.double)
flavors = [
    FullLaplace,
    DiagLaplace,
]
all_keys = [
    ("all", "full"),
    ("all", "diag"),
]


@pytest.fixture
def model():
    model = nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    return model


def test_default_init(model, likelihood="classification"):
    # test if default initialization works, id=(all, full)
    lap = Laplace(model, likelihood)
    assert isinstance(lap, FullLaplace)


@pytest.mark.parametrize("laplace, key", zip(flavors, all_keys))
def test_all_init(laplace, key, model, likelihood="classification"):
    # test if all flavors are correctly initialized
    w, s = key
    lap = Laplace(model, likelihood, subset_of_weights=w, hessian_structure=s)
    assert isinstance(lap, laplace)


@pytest.mark.parametrize("key", all_keys)
def test_opt_keywords(key, model, likelihood="classification"):
    # test if optional keywords are correctly passed on
    w, s = key
    prior_mean = torch.zeros_like(parameters_to_vector(model.parameters()))
    lap = Laplace(
        model,
        likelihood,
        subset_of_weights=w,
        hessian_structure=s,
        prior_precision=0.01,
        prior_mean=prior_mean,
        temperature=10.0,
    )
    assert torch.allclose(lap.prior_mean, prior_mean)
    assert lap.prior_precision == 0.01
    assert lap.temperature == 10.0
