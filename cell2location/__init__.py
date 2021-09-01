from pyro.distributions import constraints
from pyro.distributions.transforms import SoftplusTransform
from torch.distributions import biject_to, transform_to

from .run_c2l import run_cell2location
from .run_colocation import run_colocation
from .run_regression import run_regression

__all__ = [
    "run_cell2location",
    "run_regression",
    "run_colocation",
]


@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftplusTransform()
