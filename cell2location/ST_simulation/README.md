### Simulation of Visium spots from single-cell reference

Note: This folder provides a collection of scripts we used to simulated data but the scripts need to be edited to be used on other platforms (contains hard-coded paths for our HPC).

#### Contents

- `ST_simulation.py` functions to simulate spots
- `split_sc.py` script to split mouse brain snRNA-seq reference into generation dataset (single cells used to make the synthetic spots) and validation dataset (single cells used as a reference to train the location model)
- `assemble_ST.py` script to generate synthetic ST spots from the generation dataset

### Run simulation v2

1. Split single-cell dataset
```
python cell2location/pycell2location/ST_simulation/split_sc.py
```

2. Build design matrix (define low/high density cell types)
```
n_spots=100
seed=$(ls labels_generation* | sed 's/.*_//' | sed 's/.p//')
python cell2location/pycell2location/ST_simulation/assemble_design_2.py \
  labels_generation_${seed}.p counts_generation_${seed}.p \
  --tot_spots $n_spots
```

3. Assemble cell type composition per spot
```
id=1
python cell2location/pycell2location/ST_simulation/assemble_composition_2.py \
    labels_generation_${seed}.p counts_generation_${seed}.p \
    synthetic_ST_seed${seed}_design.csv \
    --tot_spots $n_spots --assemble_id $id
```

4. Assemble simulated ST data
```
python ${c2l_dir}/pycell2location/ST_simulation/assemble_st_2.py \
    synthetic_ST_seed${seed}_${id}_composition.csv \
    labels_generation_${seed}.p counts_generation_${seed}.p \
    --tot_spots $n_spots --assemble_id $id
```

**To simulate > 100 spots:** Define design once then run steps 2 and 3 many times using wrapper 
```
cell2location/pycell2location/ST_simulation/run_simulation2.sh $seed $n_spots $id
```
then merge in one object
```
python cell2location/pycell2location/ST_simulation/merge_synthetic_ST.py . $seed
```


