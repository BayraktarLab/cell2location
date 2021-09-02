## Common errors

### Pyro/scvi-tools version

TBC

### Pymc3 version (advanced use)

#### 1. Training cell2location on GPU takes forever (>50 hours)

1. Training cell2location using `cell2location.run_cell2location()` on GPU takes forever (>50 hours). Please check that cell2location is actually using the GPU. It is crucial to add this line in your script / notebook:

```python
# this line should go before importing cell2location
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'
import cell2location
```
which tells theano (cell2location dependency) to use the GPU before importing cell2location (or it's dependencies - theano & pymc3).
For data with 4039 locations and 10241 genes the analysis should take about 17-40 minutes depending on GPU hardware.

#### 2. `FloatingPointError: NaN occurred in optimization.`

2. `FloatingPointError: NaN occurred in optimization.` During training model parameters get into very unlikely range, resulting in division by 0 when computing gradients and breaking the optimisation:
```
FloatingPointError: NaN occurred in optimization. 
The current approximation of RV `gene_level_beta_hyp_log__`.ravel()[0] is NaN.
...
```
This usually happens when:

**A.** Numerical accuracy issues with older CUDA versions. **Solution**: use our singularity and docker images with CUDA 10.2.

**B.** The single cell reference is a very poor match to the data - reference expression signatures of cell types cannot explain most of in-situ expression. E.g. trying to map immune cell types to a tissue section that contains mostly stromal and epithelial cells. **Solution**: aim to construct a comprehensive reference.

**C.** Using cell2location in single-sample mode makes it harder to distinguish technology difference from cell abundance. **Solution**: if you have multiple expreriments try analysing them jointly in the multi-sample mode (detected automatically based on `'sample_name_col': 'sample'`).

**D.** Many genes are not expressed in the spatial data. **Solution**: try removing genes detected at low levels in spatial data.

#### 3. Theano fails to use the GPU at all (or cuDNN in particular)
3. `Can not use cuDNN on context None: cannot compile with cuDNN. ...` and other related errors. If you see these error when importing cell2location it means that you have incorrectly installed theano and it's dependencies (fix depends on the platform). Without cuDNN support training takes >3 times longer. There are **2 solutions** to this:

1. Use dockers/singularity images that are fully set up to work with the GPU (recommended).
2. Add path to system CUDA installation to the following environmental variables by adding these lines to your `.bashrc` (modify accordingly for your system):

```bash
# cuda v
cuda_v=-10.2
export CUDA_HOME=/usr/local/cuda$cuda_v
export CUDA_PATH=$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda$cuda_v/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda$cuda_v/bin:$PATH
```
