# +
# import supervised location model with co-varying cell locations
# Co-located cell type combination model
# Doesn't learn the number of factors but is fast
from .CoLocatedCombination_sklearnNMF import CoLocatedCombination_sklearnNMF
from .CoLocationModelNB4V2 import CoLocationModelNB4V2
from .LocationModelNB4V7_V4_V4 import LocationModelNB4V7_V4_V4
from .RegressionNBV2Torch import RegressionNBV2Torch
from .RegressionNBV4Torch import RegressionNBV4Torch

# pick the default model
LocationModel = CoLocationModelNB4V2
# -

__all__ = [
    "CoLocationModelNB4V2",
    "LocationModelNB4V7_V4_V4",
    "RegressionNBV2Torch",
    "RegressionNBV4Torch",
    "CoLocatedCombination_sklearnNMF",
    "LocationModel",
]
