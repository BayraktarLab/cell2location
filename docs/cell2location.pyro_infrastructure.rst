Pyro and scvi-tools infrastructure classes
==========================================

Base mixin classes (AutoGuide setup, posterior quantile computation, plotting & export)
---------------------------------------------------------------------------------------

.. automodule:: cell2location.models.base._pyro_mixin
   :members:
   :undoc-members:
   :show-inheritance:

scvi-tools Module classes (initialising the model and the guide, PyroBaseModuleClass)
-------------------------------------------------------------------------------------

Cell2location spatial cell abundance estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models.base._pyro_base_loc_module.Cell2locationBaseModule
   :members:
   :undoc-members:
   :show-inheritance:

Reference signature estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models.reference._reference_model.RegressionModel
   :members:
   :undoc-members:
   :show-inheritance:

