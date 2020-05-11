# import base classes
from .base_model import BaseModel
from .pymc3_model import Pymc3Model
from .pymc3_loc_model import Pymc3LocModel 

# +
# import supervised location models
# models with no factor-specific regularising priors
from .LocationModelV7_V4_V4 import LocationModelV7_V4_V4

# import supervised location model with NB likelihood
from .LocationModelNBV7_V4_V4 import LocationModelNBV7_V4_V4

# -

# cell neighbourhood models
from .CellNeighbourhood_sklearnNMF import CellNeighbourhood_sklearnNMF # Doesn't learn the number of factors but is very fast and similar to CellNeighbourhoodV5 when used with regularisation

# +
#from .LocationModelNBV7Torch import LocationModelNBV7Torch
#from .LocationModelV7Torch import LocationModelV7Torch
# -

# pick the default model
LocationModel = LocationModelV7_V4_V4

__all__ = [
    "BaseModel",
    "Pymc3Model",
    "Pymc3LocModel",
    
    "LocationModelV7_V4_V4",
    "LocationModelNBV7_V4_V4",
    "CellNeighbourhood_sklearnNMF",
]
