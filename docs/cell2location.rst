Cell2location: spatial mapping (scvi-tools/pyro)
================================================

User-facing cell2location spatial cell abundance estimation model class (scvi-tools BaseModelClass)
---------------------------------------------------------------------------------------------------

.. automodule:: cell2location.models.Cell2location
   :members:
   :undoc-members:
   :show-inheritance:

Pyro and scvi-tools Module classes (inc math description)
---------------------------------------------------------

Pyro Module class (defining the model using pyro, math description)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models._cell2location_module.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
   :members:
   :undoc-members:
   :show-inheritance:

scvi-tools Module class (initialising the model and the guide, PyroBaseModuleClass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models._cell2location_model.Cell2locationBaseModule
   :members:
   :undoc-members:
   :show-inheritance:

Simplified model architectures
------------------------------

No prior factorisation of w_sf (Pyro Module class, math description)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models.simplified._cell2location_v3_no_factorisation_module.LocationModelMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
   :members:
   :undoc-members:
   :show-inheritance:

No gene-specific platform effect m_g (Pyro Module class, math description)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cell2location.models.simplified._cell2location_v3_no_mg_module.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelNoMGPyroModel
   :members:
   :undoc-members:
   :show-inheritance: