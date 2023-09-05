import logging

from pyro.distributions import constraints
from pyro.distributions.transforms import SoftplusTransform
from rich.console import Console
from rich.logging import RichHandler
from torch.distributions import biject_to, transform_to

from . import models
from .cell_comm.around_target import compute_weighted_average_around_target
from .run_colocation import run_colocation

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


# define custom distribution constraints
@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftplusTransform()


package_name = "cell2location"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("cell2location: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = [
    "models",
    "run_colocation",
    "compute_weighted_average_around_target",
]
