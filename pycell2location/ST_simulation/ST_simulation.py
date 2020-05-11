import sys,os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata 
import random
import collections
import torch as t
import torch.distributions as dists

## --- Version 2: cell densities --- ##

def assemble_ct_composition(design_df, tot_spots, ncells_scale=5):
    '''
    Parameters
    ----------
    design_df: pd.DataFrame containing number of spots (nspots) and mean n of 
        cells per spot (mean_ncell) per cell type
    tot_spots: int
        total number of spots to simulate
    Return
    ------
    pd.DataFrame of cell types x spots with no of cells 
    '''
    spots_members = pd.DataFrame(columns=range(tot_spots), 
                                   index=design_df.index)
    ## Cell types to spot
    for i in range(len(design_df.nspots)):
        l = ([0] * (tot_spots - int(design_df.nspots[i]))) + ([1] * int(design_df.nspots[i]))
        l = random.sample(l, k = tot_spots)
        spots_members.iloc[i] = pd.Series(l)

    ## No of cells per spot
    ncells = [np.round(np.random.gamma(design_df.loc[ct,].mean_ncells, ncells_scale, size=int(design_df.loc[ct,].nspots))) for ct in design_df.index]
    for i in range(spots_members.shape[0]):
        spots_members.iloc[i, spots_members.columns[spots_members.iloc[i]==1]] = ncells[i]
    return(spots_members)

def assemble_spot_2(cnt, labels, members, fraction):
    uni_labels = members.index
    spot_expr = t.zeros(cnt.shape[1]).type(t.float32)
    for z in range(len(uni_labels)):
        if members[z] > 0:
            idx = np.where(labels == uni_labels[z])[0]
            # pick random cells from type
            np.random.shuffle(idx)
            idx = idx[0:int(members[z])]
            # add fraction of transcripts to spot expression
            z_expr = t.tensor((cnt.iloc[idx,:]*fraction).sum(axis = 0).round().astype(np.float32))
            spot_expr +=  z_expr
    return(spot_expr)

def assemble_st_2(cnt, labels, spots_members, gene_level_scaled):
    tot_spots = spots_members.shape[1]
    st_cnt = np.zeros((tot_spots,cnt.shape[1]))
    for spot in range(tot_spots):
        print("making spot no." + str(spot) + "...", flush=True)
        spot_data =  assemble_spot_2(cnt, labels, spots_members.iloc[:,spot], fraction=gene_level_scaled)
        st_cnt[spot,:] = spot_data
    # convert to pandas DataFrames
    index = pd.Index(['Spotx' + str(x + 1) for \
                  x in range(tot_spots) ])
    st_cnt = pd.DataFrame(st_cnt,
                          index = index,
                          columns = cnt.columns,
                         )
    return(st_cnt)



## --- Version 1: proportions --- ##

def pick_cell_types(uni_labels, alpha, min_n_cells):
    '''
    Pick cell types to include in synthetic spots with proportions from 
    Dirichlet distribution.
    
    Parameters
    ----------
    uni_labels: np.array
        unique labels
    alpha: np.array 
        dirichlet distribution concentration value 
        (can be from cell type proportions in ST)
        
    Return
    ------
    tuple of picked cell types and proportions
    
    '''
    # get number of different
    # cell types present
    n_labels = uni_labels.shape[0]
    
    # sample number of types to be present at current spot
    # w/o having more types than cells
    n_types = dists.uniform.Uniform(low = 1,
                                    high = min([n_labels, min_n_cells])).sample()

    n_types = n_types.round().type(t.int)

    # select which types to include
    pick_types = t.randperm(n_labels)[0:n_types]
    alpha = t.Tensor(np.array(alpha[pick_types]))
    
    # select cell type proportions
    member_props = dists.Dirichlet(concentration = alpha * t.ones(n_types)).sample()        
    return((pick_types, member_props))

def assemble_spot(cnt, labels, n_cells, fraction, pick_types, member_props):
    '''
    Generate one synthetic ST spot
    
    Parameters:
    -----------
    cnt: pd.DataFrame of single-cell count data --> [n_cells x n_genes] <--
    labels: pd.DataFrame of single-cell annotations [n_cells]
    n_cells: int number of cells to include in spot
    fraction: float or np.array 
        fraction of transcripts from each cell being 
        observed in ST-spot (gene budgets in model)
    pick_types: torch.Tensor of cell types to include in spot (output of pick_cell_types)
    member_props: torch.Tensor of the proportions of different cell types in spots (output of pick_cell_types)
    
    Returns:
    --------
    Dictionary with expression data,
    proportion values and number of
    cells from each type at every
    spot
    '''
    # get unique labels found in single cell data
    uni_labels, uni_counts = np.unique(labels,
                                     return_counts = True)
    n_labels = uni_labels.shape[0]

    assert np.all(uni_counts >=  30), "Insufficient number of cells"
    
    # get no. of members of spot for each cell type
    members = t.zeros(n_labels).type(t.float)
    members[pick_types] = (n_cells * member_props).round()
    # get final proportion of each type
    props = members / members.sum()
    # convert members to integers
    members = members.type(t.int)
    # generate spot expression data
    spot_expr = t.zeros(cnt.shape[1]).type(t.float32)
    nUMIs = t.zeros((len(uni_labels))).type(t.float32)
    for z in range(len(uni_labels)):
        if members[z] > 0:
            idx = np.where(labels == uni_labels[z])[0]
            # pick random cells from type
            np.random.shuffle(idx)
            idx = idx[0:members[z]]
            # add fraction of transcripts to spot expression
            z_expr = t.tensor((cnt.iloc[idx,:]*fraction).sum(axis = 0).round().astype(np.float32))
            nUMIs[z] = z_expr.sum()
            spot_expr +=  z_expr
    return {'expr':spot_expr,
            'proportions':props,
            'members': members,
            'umis': nUMIs
           }

def assemble_region(cnt, labels, n_cells_vec, alpha, fraction):
    '''
    Assemble ST-spots from a single synthetic region 
    i.e. with the same proportions of cell types in each spot

    Parameters
    ----------
    n_cell_vec: vector of number of cells to mix for each synthetic spot
    alpha: np.array 
        dirichlet distribution concentration value 
        (can be from cell type proportions in ST)
    fraction: float or np.array 
        fraction of transcripts from each cell being 
        observed in ST-spot (gene budgets in model)
    '''

    n_spots = len(n_cells_vec)

    # get unique labels
    uni_labels = np.unique(labels.values)
    n_labels = uni_labels.shape[0]

    # prepare matrices
    st_cnt = np.zeros((n_spots,cnt.shape[1]))
    st_prop = np.zeros((n_spots,n_labels))
    st_memb = np.zeros((n_spots,n_labels))
    st_umis = np.zeros((n_spots,n_labels))

    # generate one spot at a time
    pick_types,member_props = pick_cell_types(uni_labels, alpha, min(n_cells_vec))
    #     np.random.seed(1337)
    #     t.manual_seed(1337)
    for spot in range(n_spots):
        spot_data = assemble_spot(cnt,
                                 labels,
                                  n_cells_vec[spot], fraction, pick_types, member_props
                                 )

        st_cnt[spot,:] = spot_data['expr']
        st_prop[spot,:] = spot_data['proportions']
        st_memb[spot,:] =  spot_data['members']
        st_umis[spot,:] =  spot_data['umis']

        index = pd.Index(['Spotx' + str(x + 1) for \
                          x in range(n_spots) ])

    # convert to pandas DataFrames
    st_cnt = pd.DataFrame(st_cnt,
                          index = index,
                          columns = cnt.columns,
                         )
    st_prop = pd.DataFrame(st_prop,
                           index = index,
                           columns = uni_labels,
                          )
    st_memb = pd.DataFrame(st_memb,
                           index = index,
                           columns = uni_labels,
                           )
    st_umis = pd.DataFrame(st_umis,
                       index = index,
                       columns = uni_labels,
                       )
    return {'counts':st_cnt,
        'proportions':st_prop,
        'members':st_memb,
           'umis':st_umis}

def assemble_st(cnt, labels, n_regions, n_cells_tot, alpha, fraction):
    '''
    Assemble synthetic ST data from count matrix and predicted 
    cell type labels for each single-cell. Regions are modelled as groups of spots with
    the same proportion of cell types (and roughly the same number of cells per spot). 

    Parameters
    ----------
    n_spots: int 
        number of spots to simulate
    n_regions: int 
        number of regions in which spots should be divided
    alpha: np.array 
        dirichlet distribution concentration value 
        (can be from cell type proportions in ST)
    fraction: float or np.array 
        fraction of transcripts from each cell being 
        observed in ST-spot (gene budgets in model)
    
    (if you don't want zonation you can just make as many regions as spots)
    '''
    # count total number of spots
    tot_spots = len(n_cells_tot)
    
    # get unique labels
    uni_labels = np.unique(labels.values)
    n_labels = uni_labels.shape[0]
    
    # assign spots to regions
    # avoding to have regions with no spots 
    if n_regions != tot_spots:
        region_labels=[]
        while len(np.unique(region_labels))!=n_regions:
            region_labels = np.array(random.choices(range(n_regions), k=tot_spots))
    else:
        region_labels=np.array(range(n_regions))

    
    # prepare matrices
    st_cnt = np.zeros((tot_spots,cnt.shape[1]))
    st_prop = np.zeros((tot_spots,n_labels))
    st_memb = np.zeros((tot_spots,n_labels))
    st_umis = np.zeros((tot_spots,n_labels))
    idx=0
    
    # sort number of cells to have ~ same number of cells per spot for each region
    n_cells_tot.sort()
    
    # assemble one region at a time
    for reg in range(n_regions):
        print("making reg" + str(reg) + "...", flush=True)
        n_spots_reg = len(region_labels[region_labels==reg])
        n_cells_vec = n_cells_tot[idx:idx+n_spots_reg]
        reg_data = assemble_region(cnt, labels, n_cells_vec, alpha, fraction)

        st_cnt[idx:idx+n_spots_reg,:] = reg_data['counts']
        st_prop[idx:idx+n_spots_reg,:] = reg_data['proportions']
        st_memb[idx:idx+n_spots_reg,:] = reg_data['members']
        st_umis[idx:idx+n_spots_reg,:] = reg_data['umis']
        idx = idx + n_spots_reg

    index = pd.Index(['Spotx' + str(x + 1) for \
                      x in range(tot_spots) ])
    # convert to pandas DataFrames
    st_cnt = pd.DataFrame(st_cnt,
                          index = index,
                          columns = cnt.columns,
                         )

    st_prop = pd.DataFrame(st_prop,
                           index = index,
                           columns = uni_labels,
                          )
    st_memb = pd.DataFrame(st_memb,
                           index = index,
                           columns = uni_labels,
                           )
    st_umis = pd.DataFrame(st_umis,
                           index = index,
                           columns = uni_labels,
                           )
    return {'counts':st_cnt,
            'proportions':st_prop,
            'members':st_memb,
            'umis':st_umis,
            'regions':region_labels}
