### The models are defined as python classes with the following inheritance structure to allow methods to be reused:

- BaseModel - methods needed for Pymc3, Pyro and other implementations

-- Pymc3Model - methods for any Pymc3 model
--- Pymc3LocModel - methods for location models (fixed gene loadings, locating to spots)
---- LocationModelV1 - definition of one specific model: W gives nUMI because sum_g(G_fg) = 1
---- LocationModelV2 - one model: no renormalisation of G_fg after M_g scaling
---- LocationModelV3 - one model: same as 2 but the input G_fg are not proportions
---- LocationModelV4 - one model: model sample-specific scaling, & sum_g(M_g)=1

--- SpatialProgrammeV1 - one model: identifying spatial expression programmes
--- CellProgrammeV1 - one model: identifying single cell expression programmes
--- CellSpatialProgrammeV1 - one model: identifying spatial and single cell expression programmes jointly

-- PyroModel

-- CustoModel