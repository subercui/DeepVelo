# %%
import pickle
import scvelo as scv
import numpy as np
from time import time
from utils.temporal import latent_time
from utils.scatter import scatter
from utils.velocity import velocity
import os

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo', transparent=False)  # for beautified visualization
MASK_ZERO = True
DEEPVELO = False  # choice of {True, False, 'ShowTarget'}
DYNAMICAL = True  # whether use the dynamical mode of scvelo and compute latent time
DEEPVELO_FILE = 'scvelo_mat.npz'
# choice of {'EP', 'DG', 'velocyto_dg', 'velocyto_hg', 'E9M2_Glial', 'E9-11F1_Glial', 'E9-11M2_Glial', 'E9-11F1_Gluta'}
data = 'EP'
SURFIX = '[dynamical]' if DYNAMICAL else ''
SURFIX += '[deep_velo]' if DEEPVELO else ''


# %%loading and cleaningup data
if data == 'DG':
    adata = scv.datasets.dentategyrus()
elif data == 'EP':
    adata = scv.datasets.pancreas()
elif data == 'velocyto_dg':
    adata = scv.read('data/DentateGyrus.loom', cache=True)
    adata.obsm['tsne'] = adata.obs[['TSNE1', 'TSNE2']].to_numpy()
elif data == 'velocyto_hg':
    adata = scv.read('data/hgForebrainGlut.loom', cache=True)
elif data == 'E9M2_Glial':
    adata = scv.read('data/E9M2_glial.h5ad', cache=True)
elif data == 'E9-11F1_Glial':
    adata = scv.read('data/E9-11F1_glial.h5ad', cache=True)
elif data == 'E9-11M2_Glial':
    adata = scv.read('data/E9-11M2_glial.h5ad', cache=True)
elif data == 'E9-11F1_Gluta':
    adata = scv.read('data/E9-11F1_glutamergic.h5ad', cache=True)
else:
    raise ValueError(
        "choose data from \{'EP', 'DG', 'velocyto_dg', 'velocyto_hg',"
        "'E9M2_Glial', 'E9-11F1_Glial', 'E9-11M2_Glial', 'E9-11F1_Gluta'\}")


# %% Preprocessing Data
# here we have the size normalization
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

# here comes the NN graph and dynamic estimations
scv.pp.moments(adata, n_neighbors=30, n_pcs=30)


# %% Compute velocity and velocity graph
# import pudb; pudb.set_trace()
if DYNAMICAL:
    scv.tl.recover_dynamics(adata)
    velocity(adata, mode='dynamical', mask_zero=MASK_ZERO)
else:
    velocity(adata, mask_zero=MASK_ZERO)

# %% output and change the velocity
to_save = {
    'Ux_sz': adata.layers['Mu'].T,
    'Sx_sz': adata.layers['Ms'].T,
    'velo': adata.layers['velocity'].T,
    'conn': adata.obsp['connectivities'].T  # (features, cells)
}  # have to input in dimmention order (1999 genes, 2930 cells)
with open('./data/scveloDG.npz', 'wb') as f:
    pickle.dump(to_save, f)
if DEEPVELO == 'ShowTarget':
    print('computing target velocities')
    n_genes, batch_size = adata.layers['velocity'].T.shape
    from data_loader.data_loaders import VeloDataset
    ds = VeloDataset(data_dir='./data/scveloDG.npz')
    velo_mat = ds.velo.numpy()
    assert adata.layers['velocity'].shape == velo_mat.shape
    adata.layers['velocity'] = velo_mat  # (2930 cells, 1999 genes)
elif DEEPVELO:
    n_genes, batch_size = adata.layers['velocity'].T.shape
    now = time()
    os.system(
        f'python train.py -c config_figure3.json --ng {n_genes} --bs {batch_size} --ot {DEEPVELO_FILE} --dd ./data/scveloDG.npz')
    print(f'finished in {time()-now:.2f}s')

    # load
    velo_mat = np.load(f'./data/{DEEPVELO_FILE}')
    assert adata.layers['velocity'].shape == velo_mat['velo_mat'].shape
    adata.layers['velocity'] = velo_mat['velo_mat']  # (2930 cells, 1999 genes)
    # adata.layers['velocity'] = - adata.layers['velocity']
scv.tl.velocity_graph(adata)

# %% generate umap if need
if not ('X_umap' in adata.obsm or 'tsne' in adata.obsm):
    scv.tl.umap(adata)  # this will add adata.obsm: 'X_umap'

# %% plot panel a
if data == 'EP':
    scv.pl.velocity_embedding_stream(adata, basis='umap', dpi=300, save=f'figure3/velo_emb_stream{SURFIX}.png',
                                     legend_fontsize=9)

# %% panel b
scv.pl.velocity(adata, var_names=['Tmsb10', 'Ppp3ca', 'Dlg2'], basis='umap',
                dpi=300, save=f'figure3/phase_velo_exp{SURFIX}.png')
scatter(adata, var_names=['Tmsb10'], basis='umap',
        add_quiver=True, dpi=300, save=f'figure3/phase{SURFIX}1.png',
        legend_loc_lines="none")
scatter(adata, var_names=['Fam155a'], basis='umap',
        add_quiver=True, dpi=300, save=f'figure3/phase{SURFIX}2.png',
        legend_loc_lines="none")
# %% more plots
# scv.pl.velocity_graph(adata, dpi=300, save='velo_graph.pdf')
if data == 'EP':
    scv.pl.velocity(adata, var_names=['Sntg1', 'Sbspon'], basis='umap',
                    dpi=300, save=f'figure3/phase_velo_exp{SURFIX}.png')
    scatter(adata, var_names=['Sntg1', 'Sbspon'], basis='umap',
            add_quiver=True, dpi=300, save=f'figure3/phase{SURFIX}.png')
elif data.startswith('E9'):
    scv.pl.velocity(adata, var_names=['Mybl1', 'Rragb'], basis='pca',
                    dpi=300, save=f'figure3/phase_velo_exp{SURFIX}.png')
    scatter(adata, var_names=['Mybl1'], basis='pca', add_quiver=True, dpi=300, save=f'figure3/phase{SURFIX}.png')
    scatter(adata, var_names=['Rragb'], basis='pca', add_quiver=True, dpi=300, save=f'figure3/phase{SURFIX}.png')

latent_time(adata)
scv.pl.scatter(
    adata,
    color='latent_time',
    color_map='gnuplot',
    size=80,
    dpi=300,
    save=f'figure3/latent_time{SURFIX}.png'
)
