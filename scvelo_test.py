# %%
import scvelo as scv
import numpy as np
import scanpy

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo')  # for beautified visualization


# %%loading and cleaningup data
# adata = scv.datasets.dentategyrus()
adata = scv.read('./data/hgForebrainGlut.loom', cache=True)
# scanpy.pp.neighbors(adata)
# scanpy.tl.umap(adata)

# %% Preprocessing Data
# here we have the size normalization
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scanpy.tl.pca(adata)

# here comes the NN graph and dynamic estimations
scv.pp.moments(adata, n_neighbors=30, n_pcs=30)


# %% Compute velocity and velocity graph
# import pudb; pudb.set_trace()
scv.tl.velocity(adata)
# adata.layers['velocity'] = - adata.layers['velocity']
scv.tl.velocity_graph(adata)

# # %% output and change the velocity
# np.savez(
#     './data/scveloDG.npz', 
#     Ux_sz=adata.layers['Mu'].T, 
#     Sx_sz=adata.layers['Ms'].T, 
#     velo=adata.layers['velocity'].T
#     ) # have to input in dimmention order (1999 genes, 2930 cells)
# #data = np.load('./data/DG_norm_genes.npz'); data.files; data['Ux_sz']
# velo_mat = np.load('./data/velo_mat.npz')
# adata.layers['velocity'] = velo_mat['velo_mat']  # (2930 cells, 1999 genes)
# # adata.layers['velocity'] = - adata.layers['velocity']
# scv.tl.velocity_graph(adata)

# %% plot
scv.pl.velocity_embedding(adata, basis='pca', arrow_length=1.2, arrow_size=1.2, dpi=150)
# scv.pl.velocity_embedding_stream(adata, basis='umap', color=['clusters', 'age(days)'], dpi=300, show=False, save='velo_emb_stream.pdf')
scv.pl.velocity_embedding_grid(adata, basis='pca', arrow_length=1.2, arrow_size=1.2, dpi=300, show=False, save='velo_emb_grid.pdf')


# %% more plots
scv.pl.velocity_graph(adata, show=False, save='velo_graph.pdf')


# %% [markdown]
# # Have a try on the velocyto's data