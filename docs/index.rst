.. cell2location documentation master file, created by
   sphinx-quickstart on Wed Jun 17 17:03:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cell2location's documentation!
=========================================

For installation instructions see: https://github.com/BayraktarLab/cell2location#Installation

For FAQ and to ask any questions please use GitHub Discussions: https://github.com/BayraktarLab/cell2location/discussions

For reporting bugs or other issues with cell2location please use GitHub Issues: https://github.com/BayraktarLab/cell2location/issues

.. toctree::
   :maxdepth: 3
   :caption: Quick start tutorial:

   notebooks/cell2location_tutorial
   
Cell2location package is implemented in a general way (using https://pyro.ai/ and https://scvi-tools.org/) to support multiple related models - both for spatial mapping and estimating reference cell type signatures:

1. Cell2location for spatial mapping of cell types which estimates cell abundance by decomposing spatial data into reference expression signatures of cell types (`LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel`). 
2. Models for estimating reference expression signatures of cell types from scRNA data, accounting for variable sequencing depth between batches (e.g. 10X reaction), additive background (contaminating RNA), multiplicative platform effect between scRNA technologies.
3. Cell2location model for mapping to Nanostring WTA data (`LocationModelWTA`). See https://github.com/vitkl/SpaceJam for a new more versatile version.
4. Similified versions of model #1 that lack particular features of the full model, accessible from `cell2location.models.simplified`

Additionally we provide 2 models for downstream analysis of cell abundance estimates, accessible from `cell2location.models.downstream`:

1. `CoLocatedGroupsSklearnNMF` - identifying groups of cell types with similar locations using NMF (wrapper around sklearn NMF). See tutorial for usage.
2. `ArchetypalAnalysis` - identifying smoothly varying and mutually exclusive tissue zones with Archetypa Analysis.

.. toctree::
   :maxdepth: 1
   :caption: Detailed tutorials:

   notebooks/standard_workflow_from_spaceranger_to_saving_common_plots
   notebooks/downstream_analysis_advanced_plotting

.. toctree::
   :maxdepth: 2
   :caption: Using docker and singularity environments, common errors:

   dockersingularity
   commonerrors

.. toctree::
   :maxdepth: 1
   :caption: Pymc3 Tutorials (advanced use, Nanostring WTA):

   installing_pymc3
   notebooks/cell2location_for_NanostringWTA
   notebooks/cell2location_estimating_signatures
   notebooks/cell2location_short_demo
   notebooks/cell2location_short_demo_downstream

.. toctree::
   :maxdepth: 4
   :caption: API:

   cell2location.utils.filtering
   cell2location.reference_models
   cell2location.cluster_averages
   cell2location
   cell2location.downstream_models
   cell2location.plt
   cell2location.utils
   cell2location.pyro_infrastructure
   cell2location.pymc3
   cell2location.reference_models_torch
   cell2location.distributions

Indices and tables
==================

* :ref:`modindex`
* :ref:`search`
