import scvelo as scv
scv.logging.print_version()

scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.set_figure_params('scvelo')  # for beautified visualization

adata = scv.datasets.dentategyrus()

# show proportions of spliced/unspliced abundances
scv.utils.show_proportions(adata)
adata

scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

scv.tl.velocity(adata)

scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(adata, basis='umap', color=['clusters', 'age(days)'])