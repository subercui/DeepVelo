# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import sys
import numpy as np
import pandas as pd
import matplotlib
import loompy
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import scipy.optimize
import velocyto as vcy
import glob
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import pickle
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
get_ipython().system('mkdir data')


# %%
# from urllib.request import urlretrieve
# urlretrieve("http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom", "data/hgForebrainGlut.loom")


# %%
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


# %%
# Wrap implementation
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
colors20 = np.vstack((plt.cm.tab20b(np.linspace(0., 1, 20))[::2], plt.cm.tab20c(np.linspace(0, 1, 20))[1::2]))
def colormap_fun(x: np.ndarray) -> np.ndarray:
    return colors20[np.mod(x, 20)]

def add_additional_loom(vlm_base, loom_to_add, label=1):
    vlm_add = vcy.VelocytoLoom(loom_to_add)
    assert vlm_base.S.shape[0] == vlm_add.S.shape[0] # same num of genes
    num_base_cells = vlm_base.S.shape[1]
    num_add_cells = vlm_add.S.shape[1]

    #merge
    vlm_base.S = np.concatenate([vlm_base.S, vlm_add.S], axis=1)
    vlm_base.U = np.concatenate([vlm_base.U, vlm_add.U], axis=1)
    vlm_base.A = np.concatenate([vlm_base.A, vlm_add.A], axis=1)
    vlm_base.ca['CellID'] = np.concatenate(
        [vlm_base.ca['CellID'], vlm_add.ca['CellID']], axis=0)
    vlm_base.initial_cell_size = np.concatenate(
        [vlm_base.initial_cell_size, vlm_add.initial_cell_size], axis=0)
    vlm_base.initial_Ucell_size = np.concatenate(
        [vlm_base.initial_Ucell_size, vlm_add.initial_Ucell_size], axis=0)
    assert vlm_base.ca['cluster'] is not None
    vlm_base.ca['cluster'] = np.concatenate(
        [vlm_base.ca['cluster'], np.array([label]*num_add_cells, dtype=int)], axis=0)
    return vlm_base

def down_sample(vlm, max_cells):
    num_cells = vlm.S.shape[1]
    if num_cells <= max_cells:
        return vlm
    select_ids = np.random.permutation(num_cells)[:max_cells]
    vlm.S = vlm.S[:, select_ids]
    vlm.U = vlm.U[:, select_ids]
    vlm.A = vlm.A[:, select_ids]
    vlm.ca['CellID'] = vlm.ca['CellID'][select_ids]
    vlm.initial_cell_size = vlm.initial_cell_size[select_ids]
    vlm.initial_Ucell_size = vlm.initial_Ucell_size[select_ids]
    vlm.ca['cluster'] = vlm.ca['cluster'][select_ids]
    return vlm

def array_to_rmatrix(X):
    nr, nc = X.shape
    xvec = robj.FloatVector(X.transpose().reshape((X.size)))
    xr = robj.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr

def principal_curve(X, pca=True):
    """
    input : numpy.array
    returns:
    Result::Object
        Methods:
        projections - the matrix of the projectiond
        ixsort - the order ot the points (as in argsort)
        arclength - the lenght of the arc from the beginning to the point
    """
    # convert array to R matrix
    xr = array_to_rmatrix(X)
    
    if pca:
        #perform pca
        t = robj.r.prcomp(xr)
        #determine dimensionality reduction
        usedcomp = max( sum( np.array(t[t.names.index('sdev')]) > 1.1) , 4)
        usedcomp = min([usedcomp, sum( np.array(t[t.names.index('sdev')]) > 0.25), X.shape[0]])
        Xpc = np.array(t[t.names.index('x')])[:,:usedcomp]
        # convert array to R matrix
        xr = array_to_rmatrix(Xpc)

    #import the correct namespace
    princurve = importr("princurve", on_conflict='warn')
    
    #call the function
    fit1 = princurve.principal_curve(xr)
    
    #extract the outputs
    class Results:
        pass
    results = Results()
    results.projections = np.array( fit1[0] )
    results.ixsort = np.array( fit1[1] ) - 1 # R is 1 indexed
    results.arclength = np.array( fit1[2] )
    results.dist = np.array( fit1[3] )
    
    if pca:
        results.PCs = np.array(xr) #only the used components
        
    return results

# %% [markdown]
# # Load raw data

# %%
vlm = vcy.VelocytoLoom("E9F1_loom/possorted_genome_bam_0OM4Q.loom")
num_cells = vlm.S.shape[1]
vlm.ca['cluster'] = np.array([0]*num_cells, dtype=int)
vlm = add_additional_loom(vlm, "E10F1_loom/possorted_genome_bam_0DOBR.loom", label=1)
vlm = add_additional_loom(vlm, "E11F1_loom/possorted_genome_bam_983X8.loom", label=2)
# down sample
vlm = down_sample(vlm, max_cells=20000)
lineage_mask = np.zeros(len(vlm.ca['CellID']), dtype=bool)
lineage_barcodes = open('barcodes_tumor_lineage_barcodes_F_glia_barcodes.txt','r').read()
for i in range(len(lineage_mask)):
    cell_code = vlm.ca['CellID'][i][-17:-1]
    if cell_code in lineage_barcodes:
        lineage_mask[i] = True
vlm.filter_cells(lineage_mask)
vlm.colorandum = colormap_fun(vlm.ca['cluster'])
# vlm = vcy.VelocytoLoom("data/DentateGyrus.loom")
# labels = vlm.ca["Clusters"]
# manual_annotation = {str(i):[i] for i in labels}
# manual_annotation
# # Load an initial clustering (Louvein)
# labels = vlm.ca["Clusters"]
# manual_annotation = {str(i):[i] for i in labels}
# annotation_dict = {v:k for k, values in manual_annotation.items() for v in values }
# clusters = np.array([annotation_dict[i] for i in labels])
# colors20 = np.vstack((plt.cm.tab20b(np.linspace(0., 1, 20))[::2], plt.cm.tab20c(np.linspace(0, 1, 20))[1::2]))  # (20, 4) colors
# vlm.set_clusters(clusters, cluster_colors_dict={k:colors20[v[0] % 20,:] for k,v in manual_annotation.items()})

# just to find the initial cell size
vlm.normalize("S", size=True, log=False)
vlm.normalize("U", size=True,  log=False)

# these two lines delete the genes that do not express that much. So brute force; After this line U / S only has 10218 genes
vlm.score_detection_levels(min_expr_counts=30, min_cells_express=20,
                           min_expr_counts_U=0, min_cells_express_U=0)
vlm.filter_genes(by_detection_levels=True)  # they filter both on S, U with the same filter

vlm.score_cv_vs_mean(2000, plot=True, max_expr_avg=50, winsorize=True, winsor_perc=(1,99.8), svr_gamma=0.01, min_expr_cells=50)

# %% [markdown]
# # Normalizing data

# %%
vlm.filter_genes(by_cv_vs_mean=True)  # filter here again
vlm.score_detection_levels(min_expr_counts=0, min_cells_express=0,
                           min_expr_counts_U=25, min_cells_express_U=20)
# vlm.score_cluster_expression(min_avg_U=0.007, min_avg_S=0.06)
# vlm.filter_genes(by_detection_levels=True, by_cluster_expression=True)
# %% pca
vlm.normalize_by_total(min_perc_U=0.5)
vlm.adjust_totS_totU(normalize_total=True, fit_with_low_U=False, svr_C=1, svr_gamma=1e-04) # denoise again, similar to filter the expression

vlm.perform_PCA()
plt.plot(np.cumsum(vlm.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(vlm.pca.explained_variance_ratio_))>0.0055))[0][0]
vlm.pcs[:,1] *= -1 # flip for consistency with previous version

# %% unsupervised clustering
from sklearn.neighbors import NearestNeighbors
import igraph
nn = NearestNeighbors(50)
nn.fit(vlm.pcs[:,:4])
knn_pca = nn.kneighbors_graph(mode='distance')
knn_pca = knn_pca.tocoo()
G = igraph.Graph(list(zip(knn_pca.row, knn_pca.col)), directed=False, edge_attrs={'weight': knn_pca.data})
VxCl = G.community_multilevel(return_levels=False, weights="weight")
labels = np.array(VxCl.membership)

from numpy_groupies import aggregate, aggregate_np
pc_obj = principal_curve(vlm.pcs[:,:4], False)
pc_obj.arclength = np.max(pc_obj.arclength) - pc_obj.arclength  # transfer from distance to similarity
labels = np.argsort(np.argsort(aggregate_np(labels, pc_obj.arclength, func=np.median)))[labels]

# here is the clustering
manual_annotation = {str(i):[i] for i in labels}
annotation_dict = {v:k for k, values in manual_annotation.items() for v in values }
clusters = np.array([annotation_dict[i] for i in labels])
colors20 = np.vstack((plt.cm.tab20b(np.linspace(0., 1, 20))[::2], plt.cm.tab20c(np.linspace(0, 1, 20))[1::2]))  # this is just setting colors
vlm.set_clusters(clusters, cluster_colors_dict={k:colors20[v[0] % 20,:] for k,v in manual_annotation.items()})

#%% knn imputation
k = 550
# you have a knn computing from some pca
# and now you are applying some knn smoothing on the data matrix based on that.
vlm.knn_imputation(n_pca_dims=n_comps,k=k, balanced=True,
                   b_sight=np.minimum(k*8, vlm.S.shape[1]-1),
                   b_maxl=np.minimum(k*4, vlm.S.shape[1]-1))

vlm.normalize_median()
vlm.fit_gammas(maxmin_perc=[2,95], limit_gamma=True)

vlm.normalize(which="imputed", size=False, log=True)
vlm.Pcs = np.array(vlm.pcs[:,:2], order="C")  # the first two axis of the pca

# %% [markdown]
# ## 注意改成constant_unspliced之后，右上角的velocity方向就明显有估计不准的

# %%
# here we have the velocity and all the normalized S and U. we should wrap this to form a dataset for the pytorch.
# the S matrix normalized here is vlm.Sx_sz; the U matrix normalized here is vlm.Ux_sz; and the Upred is S*gamma
# and stores the U - S*gamma in a np array - vlm.velocity
# So we could just store the Ux_sz, the Sx_sz and the velocity

# try on "constant_unspliced assumption"
vlm.predict_U()  # basically gamma * S

# velocity is Ux - Upred = Ux - gamma * S_x
# and Ux is a normalized U, so the velocity is the difference from the real current U to the future predict U
vlm.calculate_velocity()  # velocity is Ux - Upred = Ux - gamma * S_x

np.savez('./data/DG_norm_genes.npz', Ux_sz=vlm.Ux_sz, Sx_sz=vlm.Sx_sz, velo=vlm.velocity)
#data = np.load('./data/DG_norm_genes.npz'); data.files; data['Ux_sz']
# velo_mat = np.load('./data/velo_mat.npz')
# vlm.velocity = velo_mat['velo_mat'].T  # (1448, 1720)

# %% plot the velocity
if not 'colorandum' in dir(vlm):
    vlm.colorandum = None
vlm.calculate_shift(assumption="constant_velocity")  # the numerical integration step, but basically the velocity
vlm.extrapolate_cell_at_t(delta_t=1)  # calculate this one and then just have a look which one it looks like


kw = dict(prop="colors", fmt="E{x:02d}", func=lambda c: c.astype(int)+9)

vlm.estimate_transition_prob(hidim="Sx_sz", embed="Pcs", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(25, 25), n_neighbors=200)

plt.figure(None,(9,9), dpi=300)
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.6, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True, "c":vlm.ca['cluster'], "cmap":"Paired"},
                     min_mass=2.9, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.8, quiver_scale=0.3, scale_type="absolute")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal");
scatter = plt.findobj(match=PathCollection)[0]
plt.legend(*scatter.legend_elements(**kw))
plt.title("E9-11F1_pca_velocity")
plt.tight_layout()
plt.savefig("E9-11F1_pca_velocity.png")

# %% [markdown]
# # tsne plot

# %%
from sklearn.manifold import TSNE
bh_tsne = TSNE()
vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])
# %%
vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(36, 36), n_neighbors=200)

plt.figure(None,(12,12),dpi=300)
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True, "c":vlm.ca['cluster'], "cmap":"Paired"},
                     min_mass=2.9, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.2, quiver_scale=0.22, scale_type="absolute")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal");
scatter = plt.findobj(match=PathCollection)[0]
plt.legend(*scatter.legend_elements(**kw))
plt.tight_layout()
plt.title("E9-11F1_tsne_velocity")
plt.savefig("E9-11F1_tsne_velocity.png")


# %%
vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)
vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=True)


# %%
vlm.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)
plt.figure(None,(20,10))
vlm.plot_grid_arrows(quiver_scale=0.6,
                    scatter_kwargs_dict={"alpha":0.35, "lw":0.35, "edgecolor":"0.4", "s":38, "rasterized":True}, min_mass=24, angles='xy', scale_units='xy',
                    headaxislength=2.75, headlength=5, headwidth=4.8, minlength=1.5,
                    plot_random=True, scale_type="absolute")


# %% umap
import umap
umapper = umap.UMAP(
    n_neighbors=60,  # 15
    min_dist=0.6,  # 0.1
)
vlm.umap = umapper.fit_transform(vlm.pcs[:, :25])


vlm.estimate_transition_prob(hidim="Sx_sz", embed="umap", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(36, 36), n_neighbors=200)

plt.figure(None,(12,12),dpi=300)
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True, "c":vlm.ca['cluster'], "cmap":"Paired"},
                     min_mass=2.9, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.2, quiver_scale=0.9, scale_type="absolute")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal");
scatter = plt.findobj(match=PathCollection)[0]
plt.legend(*scatter.legend_elements(**kw))
plt.title("E9-11F1_umap_velocity")
plt.tight_layout()
plt.savefig("E9-11F1_umap_velocity.png")

