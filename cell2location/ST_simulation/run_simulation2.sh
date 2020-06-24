#!/bin/bash

conda activate pymc-2

seed=$1
n_spots=$2
id=$3
c2l_dir=$4

python ${c2l_dir}/cell2location/ST_simulation/assemble_composition_2.py \
    labels_generation_${seed}.p counts_generation_${seed}.p \
    synthetic_ST_seed${seed}_design.csv \
    --tot_spots $n_spots --assemble_id $id
    
python ${c2l_dir}/cell2location/ST_simulation/assemble_st_2.py \
    labels_generation_${seed}.p counts_generation_${seed}.p \
    synthetic_ST_seed${seed}_${id}_composition.csv \
    --assemble_id $id
