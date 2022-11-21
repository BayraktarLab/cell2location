from pyro.distributions import constraints
from pyro.distributions.transforms import SoftplusTransform
from torch.distributions import biject_to, transform_to

from . import models
from .run_colocation import run_colocation

__all__ = [
    "models",
    "run_colocation",
]


@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftplusTransform()
