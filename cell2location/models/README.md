<!-- #region -->
### The models are defined as python classes with the following inheritance structure to allow methods to be reused:

'- BaseModel - methods needed for Pymc3, Torch and other implementations

'-- Pymc3Model - methods for any Pymc3 model

'--- Pymc3LocModel - methods for location models (fixed gene loadings (columns), inferring weights for locations (rows))

'---- CoLocationModelNB4V2 - 

'---- LocationModelNB4V7_V4_V4

  
  

'-- CoLocatedCombination_sklearnNMF

  
  

'-- TorchModel

'--- RegressionTorchModel

'---- RegressionNBV2Torch

'---- RegressionNBV4Torch

<!-- #endregion -->

```python

```
