# CLAUDE.md - Project Guidelines for Claude Code

## Project Overview

cell2location is a Bayesian model for spatial transcriptomics that estimates cell type abundances in spatial data by integrating single-cell RNA-seq reference signatures. It is built on top of scvi-tools and Pyro.

Repository: https://github.com/BayraktarLab/cell2location

## Tech Stack

- **Language**: Python 3.10+
- **Core dependencies**: pyro-ppl, scvi-tools, torch, numpy, pandas, scanpy
- **ML framework**: PyTorch + Pyro (probabilistic programming)
- **Data format**: AnnData (h5ad files)

## Code Style & Formatting

- **Black** for code formatting (line length: 120)
- **isort** for import sorting (profile: black, trailing comma)
- **flake8** for linting (ignored: E203, E266, E501, W503, W605, N812; max line length: 119)
- Pre-commit hooks enforce all of the above

Run formatting before committing:
```bash
black --line-length 120 .
isort .
flake8
```

## Project Structure

```
cell2location/
├── models/           # Main model classes (Cell2location, RegressionModel)
│   ├── base/         # Base modules and mixins (Pyro integration)
│   ├── reference/    # Reference signature estimation model
│   └── simplified/   # Simplified model variants
├── nn/               # Neural network layers (FC layers, context layers)
├── dataloaders/      # Custom data loaders for spatial data
├── distributions/    # Custom probability distributions
├── cell_comm/        # Cell communication analysis
├── cluster_averages/ # Cluster average computation
├── plt/              # Plotting utilities
└── utils/            # General utilities
```

## Testing

Tests are in `tests/`. Run with:
```bash
pytest
```

CI runs on Python 3.10 (Ubuntu). Tests include model training smoke tests.

## Key Architecture Patterns

- Models follow the scvi-tools pattern: a user-facing `Model` class wraps a PyTorch `Module`
- Pyro is used for variational inference (SVI with ELBO)
- `QuantileMixin` and `PltExportMixin` add posterior quantile computation and plotting
- Data registration uses scvi-tools `AnnDataManager` with field types (LayerField, CategoricalObsField, etc.)

## Important Notes

- Always ensure `adata.var_names` matches `cell_state_df.index` when using Cell2location
- The `N_cells_per_location` parameter significantly affects model behavior
- Detection mean correction handles sensitivity differences between spatial and single-cell data
