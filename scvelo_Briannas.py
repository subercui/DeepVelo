# %%
import scvelo as scv
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import os

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo', transparent=False)  # for beautified visualization
DEEPVELO = False
FILTER_CELL = True  # whether only include the subset of cells in clusters from Brianna
LOAD_EMBS = False  # whether use the precomputed embeddings
DYNAMICAL = False  # whether use the dynamical mode of scvelo and compute latent time
DEEPVELO_FILE = 'scvelo_mat.npz'
MODE = 'CF'  # choices {"combined", "CF", "TM"}

PREFIX = f'Brianna_Filtered_{MODE}_' if FILTER_CELL else f'Brianna_AllType_{MODE}_'
SURFIX = '[load_embs]' if FILTER_CELL and LOAD_EMBS else ''
SURFIX += '[dynamical]' if DYNAMICAL else ''
SURFIX += '[deep_velo]' if DEEPVELO else ''


# %%loading and cleaningup data
# adata = scv.read('data/FromBrianna/CF-4014.loom', cache=True)
if MODE == 'combined':
    adatas = {
        "CF-0404": scv.read('data/FromBrianna/CF-0404.loom', cache=True),
        "CF-19301":scv.read('data/FromBrianna/CF-19301.loom', cache=True),
        "CF-2797":scv.read('data/FromBrianna/CF-2797.loom', cache=True),
        "CF-318-813":scv.read('data/FromBrianna/CF-318-813.loom', cache=True),
        "CF-428-112":scv.read('data/FromBrianna/CF-428-112.loom', cache=True),
        "CF-7780":scv.read('data/FromBrianna/CF-7780.loom', cache=True),
        "TM-2768":scv.read('data/FromBrianna/TM-2768.loom', cache=True),
        "TM-3937":scv.read('data/FromBrianna/TM-3937.loom', cache=True),
        "TM-6477":scv.read('data/FromBrianna/TM-6477.loom', cache=True),
        "TM-7567":scv.read('data/FromBrianna/TM-7567.loom', cache=True),
        "TM-8249":scv.read('data/FromBrianna/TM-8249.loom', cache=True),
        "TM-9469":scv.read('data/FromBrianna/TM-9469.loom', cache=True),
        "TM-9817":scv.read('data/FromBrianna/TM-9817.loom', cache=True)
    }
    n_neighbors, n_pcs = 30, 30
elif MODE == 'CF':
    adatas = {
        "CF-0404": scv.read('data/FromBrianna/CF-0404.loom', cache=True),
        "CF-19301":scv.read('data/FromBrianna/CF-19301.loom', cache=True),
        "CF-2797":scv.read('data/FromBrianna/CF-2797.loom', cache=True),
        "CF-318-813":scv.read('data/FromBrianna/CF-318-813.loom', cache=True),
        "CF-428-112":scv.read('data/FromBrianna/CF-428-112.loom', cache=True)
    }
    n_neighbors, n_pcs = 30, 30
elif MODE == 'TM':
    adatas = {
        "TM-2768":scv.read('data/FromBrianna/TM-2768.loom', cache=True),
        "TM-3937":scv.read('data/FromBrianna/TM-3937.loom', cache=True),
        "TM-6477":scv.read('data/FromBrianna/TM-6477.loom', cache=True),
        "TM-7567":scv.read('data/FromBrianna/TM-7567.loom', cache=True),
        "TM-8249":scv.read('data/FromBrianna/TM-8249.loom', cache=True),
        "TM-9469":scv.read('data/FromBrianna/TM-9469.loom', cache=True),
        "TM-9817":scv.read('data/FromBrianna/TM-9817.loom', cache=True)
    }
    n_neighbors, n_pcs = 30, 30
# concatenate approach 1 - ad.concat
for k, v in adatas.items():
    v.var_names_make_unique()
adata = ad.concat(adatas, join='outer', label="dataset")
# TODO(Haotian): concatenate approach 2 - scv.utils.merge

# %% add clusters and embeddings
# clusters
cluster_file = 'data/FromBrianna/LUM_Pos_Clusters_Aug4.csv'
clusters_df = pd.read_csv(cluster_file)
clusters_dict = clusters_df.set_index('ID').to_dict()['Res.0.2_Added']
clusters_dict = {k[:16]:v for k,v in clusters_dict.items()}

cells = adata.obs.index.to_list()
clusters = []
cell_mask = []
for i, cell in enumerate(cells):
    cell_ID = cell[-17:-1]
    if cell_ID in clusters_dict:
        clusters.append(clusters_dict[cell_ID])
        cell_mask.append(True)
    else:
        # print(f'the {i}-th cell, ID {cell_ID} not found in adata, set cluster to -1.')
        clusters.append(-1)
        cell_mask.append(False)
assert len(cell_mask) == len(cells)
cell_mask = np.array(cell_mask, dtype=bool)
adata.obs['clusters'] = clusters

# filter cells acording to known cell_IDs in clusters
if FILTER_CELL:
    assert len(adata.uns) == 0
    assert len(adata.obsm) == 0
    assert len(adata.varm) == 0
    adata = ad.AnnData(
        X=adata.X[cell_mask],
        obs=adata.obs[cell_mask],
        var=adata.var,
        layers={k:v[cell_mask] for k,v in adata.layers.items()}
    )
# embeddings
if FILTER_CELL and LOAD_EMBS:
    tmp_adata = sc.read_h5ad('data/FromBrianna/Harmony_Batch_LumPos_Res02Added.h5ad')
    embs_dict = {ID[:16]:tmp_adata.obsm['X_umap'][i] for i, ID in enumerate(tmp_adata.obs.index)}
    del tmp_adata
    embs = []
    for i, cell in enumerate(adata.obs.index):
        cell_ID = cell[-17:-1]
        if cell_ID in embs_dict:
            embs.append(embs_dict[cell_ID])
        else:
            raise Exception('cell embs not found in embedding file')
    adata.obsm['X_umap'] = np.array(embs)
# adata.write('data/FromBrianna/raw_filtered_Brianna_adata.h5ad', compression='gzip')


# %% Preprocessing Data
# here we have the size normalization
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

# here comes the NN graph and dynamic estimations
scv.pp.moments(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)


# %% Compute velocity and velocity graph
# import pudb; pudb.set_trace()
if DYNAMICAL:
    scv.tl.recover_dynamics(adata)
    scv.tl.velocity(adata, mode='dynamical')
else:
    scv.tl.velocity(adata)

# %% output and change the velocity
np.savez(
    './data/scveloDG.npz', 
    Ux_sz=adata.layers['Mu'].T, 
    Sx_sz=adata.layers['Ms'].T, 
    velo=adata.layers['velocity'].T
    ) # have to input in dimmention order (1999 genes, 2930 cells)
#data = np.load('./data/DG_norm_genes.npz'); data.files; data['Ux_sz']
if DEEPVELO:
    n_genes, batch_size = adata.layers['velocity'].T.shape
    # os.system(f'python train.py -c config.json --ng {n_genes} --bs {batch_size} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # # if using self attention
    # os.system(f'python train.py -c config_SelfAttention.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # # if using base model
    # os.system(f'python train.py -c config_BaseModel.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # if using test
    os.system(f'python train.py -c config_test.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # os.system(f'python train.py -c config_test_gcn.json --ng {n_genes} --bs {batch_size} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')

    # load
    velo_mat = np.load(f'./data/{DEEPVELO_FILE}')
    assert adata.layers['velocity'].shape == velo_mat['velo_mat'].shape
    adata.layers['velocity'] = velo_mat['velo_mat']  # (2930 cells, 1999 genes)
    # adata.layers['velocity'] = - adata.layers['velocity']
scv.tl.velocity_graph(adata)

# %% generate umap if need
if not 'X_umap' in adata.obsm:
    scv.tl.umap(adata)  # this will add adata.obsm: 'X_umap'

# %% store data
# adata.write('data/FromBrianna/processed_Brianna_adata.h5ad', compression='gzip')

# %% plot
scv.pl.velocity_embedding_stream(
    adata, 
    basis='umap', 
    color='clusters', 
    dpi=300, 
    save=f'{PREFIX}velo_emb_stream{SURFIX}.png'
    # title=f'{PREFIX}velocity{SURFIX}',
)

scv.pl.velocity_embedding_stream(
    adata, 
    basis='umap', 
    color='dataset', 
    title='sample sources',
    dpi=300, 
    save=f'{PREFIX}dataset{SURFIX}.png'
)

# %% more plots
# scv.pl.velocity_graph(adata, dpi=300, save=f'{PREFIX}velo_graph.png')

if DYNAMICAL:
    scv.tl.latent_time(adata)
    scv.pl.scatter(
        adata, 
        color='latent_time', 
        color_map='gnuplot', 
        size=80,
        dpi=300,
        save=f'{PREFIX}latent_time{SURFIX}.png'
    )