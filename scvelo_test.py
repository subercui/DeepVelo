# %%
import scvelo as scv
import numpy as np
from time import time
import os

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo', transparent=False)  # for beautified visualization
DEEPVELO = True
DYNAMICAL = False  # whether use the dynamical mode of scvelo and compute latent time
DEEPVELO_FILE = 'scvelo_mat.npz'
data = 'EP'
SURFIX = '[dynamical]' if DYNAMICAL else ''
SURFIX += '[deep_velo]' if DEEPVELO else ''


# %%loading and cleaningup data
if data == 'DG':
    adata = scv.datasets.dentategyrus()
elif data == 'EP':
    adata = scv.datasets.pancreas()


# %% Preprocessing Data
# here we have the size normalization
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

# here comes the NN graph and dynamic estimations
scv.pp.moments(adata, n_neighbors=30, n_pcs=30)


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
    now = time()
    # os.system(f'python train.py -c config.json --ng {n_genes} --bs {batch_size} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # # if using self attention
    # os.system(f'python train.py -c config_SelfAttention.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # # if using base model
    # os.system(f'python train.py -c config_BaseModel.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    # if using test
    # os.system(f'python train.py -c config_test.json --ng {n_genes} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    os.system(f'python train.py -c config_test_gcn.json --ng {n_genes} --bs {batch_size} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    print(f'finished in {time()-now:.2f}s')

    # load
    velo_mat = np.load(f'./data/{DEEPVELO_FILE}')
    assert adata.layers['velocity'].shape == velo_mat['velo_mat'].shape
    adata.layers['velocity'] = velo_mat['velo_mat']  # (2930 cells, 1999 genes)
    # adata.layers['velocity'] = - adata.layers['velocity']
scv.tl.velocity_graph(adata)

# %% plot
if data == 'DG':
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=['clusters', 'age(days)'], dpi=300, save=f'velo_emb_stream{SURFIX}.png')
    # scv.pl.velocity_embedding(adata, basis='umap', arrow_length=1.2, arrow_size=1.2, dpi=150)
    scv.pl.velocity_embedding_grid(adata, basis='umap', arrow_length=1.2, arrow_size=1.2, dpi=300, save=f'velo_emb_grid{SURFIX}.png')
elif data == 'EP':
    scv.pl.velocity_embedding_stream(adata, basis='umap', dpi=300, save=f'velo_emb_stream{SURFIX}.png')

# %% more plots
# scv.pl.velocity_graph(adata, dpi=300, save='velo_graph.pdf')
scv.pl.velocity(adata, var_names=['Sntg1', 'Sbspon'], basis='umap', dpi=300, save=f'phase_velo_exp{SURFIX}.png')
scv.pl.scatter(adata, var_names=['Sntg1', 'Sbspon'], basis='umap', dpi=300, save=f'phase{SURFIX}.png')
if DYNAMICAL:
    scv.tl.latent_time(adata)
    scv.pl.scatter(
        adata, 
        color='latent_time', 
        color_map='gnuplot', 
        size=80,
        dpi=300,
        save=f'latent_time{SURFIX}.png'
    )


# %% [markdown]
# # Have a try on the velocyto's data

# %% Try to run evaluations

# prepare cluster labels
all_labels = adata.obs.clusters.to_numpy()
list_granule_immature = all_labels == 'Granule immature'
list_granule_mature = all_labels == 'Granule mature'
velos_grabule_immature = adata.layers['velocity'][list_granule_immature]
velos_grabule_mature = adata.layers['velocity'][list_granule_mature]

# get the average velocity
ave_velo_grabule_immature = velos_grabule_immature.mean(0)  # (1999,)
ave_velo_grabule_mature = velos_grabule_mature.mean(0)  # (1999,)

# metric 1: relative variance of cosine similarity within cluster
from sklearn import preprocessing
def metric1(data):
    norm_data = preprocessing.normalize(data, axis=1)
    var_cos = (norm_data@norm_data.T).var()
    return var_cos
var_velo_grabule_immature = metric1(velos_grabule_immature)
var_velo_grabule_mature = metric1(velos_grabule_mature)
print(f"metric 1:")
print(f"var of cosine similarity with in Grabule Immature Cells: {var_velo_grabule_immature}")
print(f"var of cosine similarity with in Grabule mature Cells: {var_velo_grabule_mature}")

# metric 2:
# %%
