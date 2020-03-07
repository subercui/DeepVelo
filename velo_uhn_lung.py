# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
import time
import csv
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
from os.path import join
import pickle


# %%
savedir = join('figures', time.strftime('%h %d-%H:%M'))
if not os.path.exists(savedir):
    os.mkdir(savedir)
try:
    os.mkdir('data')
except:
    pass


# %%
# from urllib.request import urlretrieve
# urlretrieve("http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom", "data/hgForebrainGlut.loom")


# %%
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


# %%
# Wrap implementation

def make_filterarrays(ds):
    with open('64_lc_leiden_barcode_cluster.tsv', 'r') as f:
        reader = csv.reader(f, dialect='excel-tab') 
        reader.__next__()
        pos_cells = []
        clusters = []
        for row in reader:
            pos_cells.append(row[0].split('.')[0])
            clusters.append(int(row[1]))

    cell_select = np.zeros(ds.ca.CellID.shape[0], dtype=np.bool)
    _clusters = np.zeros(ds.ca.CellID.shape[0], dtype='int64')
    for i in range(len(pos_cells)):
        cell = pos_cells[i]
        for j in range(ds.ca.CellID.shape[0]):
            if cell in ds.ca.CellID[j]:
                cell_select[j] = True
                _clusters[j] = clusters[i]
                continue
    # for cell in pos_cells:
    #     cell_select = np.logical_or(
    #         cell_select, 
    #         [cell in id for id in ds.ca.CellID]
    #         )
    gene_select = np.array(['mm10_' not in id for id in ds.ra.Gene])
    return cell_select, gene_select, _clusters

import rpy2.robjects as robj
from rpy2.robjects.packages import importr

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
# # Load raw data and filter human cells and genes

# %%
# vlm = vcy.VelocytoLoom("data/hgForebrainGlut.loom")
# vlm = vcy.VelocytoLoom("data/DentateGyrus.loom")

ds = loompy.connect("/cluster/projects/bwanggroup/for_haotian/velocyto/50_sample_concat/velocyto/50_sample_concat.loom")
cell_select, gene_select, _clusters = make_filterarrays(ds)
ds.close(); del ds
vlm = vcy.VelocytoLoom("/cluster/projects/bwanggroup/for_haotian/velocyto/50_sample_concat/velocyto/50_sample_concat.loom")
vlm.ca["Clusters"] = np.asarray(_clusters, dtype='int64')
vlm.filter_cells(cell_select)
vlm.filter_genes(by_custom_array=gene_select)
print(f'shape after filtering: {vlm.S.shape}')

# %% preprocessing
# Load an initial clustering (Louvein)
labels = vlm.ca["Clusters"]
manual_annotation = {str(i):[i] for i in labels}
annotation_dict = {v:k for k, values in manual_annotation.items() for v in values }
clusters = np.array([annotation_dict[i] for i in labels])
colors20 = np.vstack((plt.cm.tab20b(np.linspace(0., 1, 20))[::2], plt.cm.tab20c(np.linspace(0, 1, 20))[1::2]))  # (20, 4) colors
vlm.set_clusters(clusters, cluster_colors_dict={k:colors20[v[0] % 20,:] for k,v in manual_annotation.items()})

# just to find the initial cell size
vlm.normalize("S", size=True, log=False)
vlm.normalize("U", size=True,  log=False)
# %% [markdown]
# # Normalizing data

# %%
# these two lines delete the genes that do not express that much. So brute force; After this line U / S only has 10218 genes
vlm.score_detection_levels(min_expr_counts=15, min_cells_express=10,
                           min_expr_counts_U=15, min_cells_express_U=10)
vlm.detection_level_selected.mean()
vlm.filter_genes(by_detection_levels=True)  # they filter both on S, U with the same filter

vlm.score_cv_vs_mean(2000, plot=True, max_expr_avg=50, winsorize=True, winsor_perc=(1,99.8), svr_gamma=0.01, min_expr_cells=50)
plt.savefig(join(savedir, "score_cv.png"))
vlm.filter_genes(by_cv_vs_mean=True)  # filter here again


# vlm.score_detection_levels(min_expr_counts=0, min_cells_express=0,
#                            min_expr_counts_U=25, min_cells_express_U=20)
# vlm.score_cluster_expression(min_avg_U=0.007, min_avg_S=0.06)
# vlm.filter_genes(by_detection_levels=True, by_cluster_expression=True)
vlm.normalize_by_total(min_perc_U=0.5)
vlm.adjust_totS_totU(normalize_total=True, fit_with_low_U=False, svr_C=1, svr_gamma=1e-04) # denoise again, similar to filter the expression

vlm.perform_PCA()
plt.plot(np.cumsum(vlm.pca.explained_variance_ratio_)[:100])
plt.savefig(join(savedir, "cum_pca.png"))
n_comps = np.where(np.diff(np.diff(np.cumsum(vlm.pca.explained_variance_ratio_))>0.0055))[0][0]
vlm.pcs[:,1] *= -1 # flip for consistency with previous version

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

# np.savez('./data/DG_norm_genes.npz', Ux_sz=vlm.Ux_sz, Sx_sz=vlm.Sx_sz, velo=vlm.velocity)
#data = np.load('./data/DG_norm_genes.npz'); data.files; data['Ux_sz']
# velo_mat = np.load('./data/velo_mat.npz')
# vlm.velocity = velo_mat['velo_mat'].T  # (1448, 1720)

# %% plot the velocity
vlm.calculate_shift(assumption="constant_velocity")  # the numerical integration step, but basically the velocity
vlm.extrapolate_cell_at_t(delta_t=1)  # calculate this one and then just have a look which one it looks like

vlm.estimate_transition_prob(hidim="Sx_sz", embed="Pcs", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(25, 25), n_neighbors=200)

plt.figure(None,(9,9))
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True, "c":labels},
                     min_mass=2.7, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.8, quiver_scale=0.6, scale_type="relative")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal")
scatter = plt.findobj(match=PathCollection)[0]
# scatter.set_array(clusters.astype('int64'))
plt.legend(*scatter.legend_elements())
plt.savefig(join(savedir, "pca_plot.png"))

# %% [markdown]
# # umap plot

# %%
from umap import UMAP
bh_umap = UMAP()
vlm.umap = bh_umap.fit_transform(vlm.Sx_sz.T)
vlm.estimate_transition_prob(hidim="Sx_sz", embed="umap", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(36, 36), n_neighbors=200)

plt.figure(None,(6,6),dpi=150)
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":40, "rasterized":True, "c":labels},
                     min_mass=2.7, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.2, quiver_scale=0.6, scale_type="relative")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal");
scatter = plt.findobj(match=PathCollection)[0]
plt.legend(*scatter.legend_elements())
plt.tight_layout()
plt.savefig(join(savedir, "umap_plot.png"))

# %% [markdown]
# # tsne plot

# %%
from sklearn.manifold import TSNE
bh_tsne = TSNE()
vlm.ts = bh_tsne.fit_transform(vlm.Sx_sz.T)
vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="log", psc=1,
                             n_neighbors=150, knn_random=True, sampled_fraction=1)  # what it is doing with this one?! - compute the correlation coefficient

vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
vlm.calculate_grid_arrows(smooth=0.9, steps=(36, 36), n_neighbors=200)

plt.figure(None,(9,9))
vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":40, "rasterized":True, "c":labels},
                     min_mass=2.4, angles='xy', scale_units='xy',
                     headaxislength=2.75, headlength=5, headwidth=4.2, quiver_scale=0.6, scale_type="relative")
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
# plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
plt.gca().invert_xaxis()
plt.axis("off")
plt.axis("equal");
scatter = plt.findobj(match=PathCollection)[0]
plt.legend(*scatter.legend_elements())
plt.tight_layout()
plt.savefig(join(savedir, "tsne_plot.png"))


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


# %%


