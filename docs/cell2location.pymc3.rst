Pymc3 implementation (advanced use)
===================================

Pipelines (wrappers for full workflow)
--------------------------------------

Run cell2location
^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.run_c2l
   :members:
   :undoc-members:
   :show-inheritance:

Run regression
^^^^^^^^^^^^^^

.. automodule:: cell2location.run_regression
   :members:
   :undoc-members:
   :show-inheritance:

Main models: general and Nanostring WTA
---------------------------------------

Pymc3 (main cell2location model): LocationModelLinearDependentWMultiExperimentLocationBackgroundNormGeneAlpha
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormGeneAlpha
   :members:
   :undoc-members:
   :show-inheritance:

Nanostring WTA model: LocationModelWTA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.LocationModelWTA
   :members:
   :undoc-members:
   :show-inheritance:

Models with simplified architecture
-----------------------------------

No normalisation: LocationModelLinearDependentWMultiExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.simplified.LocationModelLinearDependentWMultiExperiment
   :members:
   :undoc-members:
   :show-inheritance:

No prior factorisation of w_sf (but hierarchical priors): LocationModelHierarchicalWMultiExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.simplified.LocationModelHierarchicalWMultiExperiment
   :members:
   :undoc-members:
   :show-inheritance:

No prior factorisation of w_sf: LocationModelMultiExperiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.simplified.LocationModelMultiExperiment
   :members:
   :undoc-members:
   :show-inheritance:

No gene-specific platform effect m_g: LocationModelLinearDependentWMultiExperimentNoMg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.simplified.LocationModelLinearDependentWMultiExperimentNoMg
   :members:
   :undoc-members:
   :show-inheritance:

No additive background RNA: LocationModelLinearDependentWMultiExperimentNoSegLs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.pymc3.simplified.LocationModelLinearDependentWMultiExperimentNoSegLs
   :members:
   :undoc-members:
   :show-inheritance:

Base model classes (infrastructure)
-----------------------------------

Pymc3: BaseModel
^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.base.base_model
   :members:
   :undoc-members:
   :show-inheritance:

Pymc3: Pymc3LocModel
^^^^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.base.pymc3_loc_model
   :members:
   :undoc-members:
   :show-inheritance:

Pymc3: Pymc3Model
^^^^^^^^^^^^^^^^^

.. automodule:: cell2location.models.base.pymc3_model
   :members:
   :undoc-members:
   :show-inheritance: