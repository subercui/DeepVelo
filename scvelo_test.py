# %%
import scvelo as scv

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.set_figure_params('scvelo')  # for beautified visualization


# %%loading and cleaningup data
adata = scv.datasets.dentategyrus()


# %% Preprocessing Data
# here we have the size normalization
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

# here comes the NN graph and dynamic estimations
scv.pp.moments(adata, n_neighbors=30, n_pcs=30)


# %% Compute velocity and velocity graph
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)


# %% plot
scv.pl.velocity_embedding_stream(adata, basis='umap', color=['clusters', 'age(days)'])
scv.pl.velocity_embedding(adata, basis='umap', arrow_length=1.2, arrow_size=1.2, dpi=150)

