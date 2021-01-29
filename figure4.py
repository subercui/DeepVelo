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
DEEPVELO = True  # choice of {True, False, 'ShowTarget'}
DYNAMICAL = False  # whether use the dynamical mode of scvelo and compute latent time
DEEPVELO_FILE = 'scvelo_mat.npz'
# choice of {'EP', 'DG', 'velocyto_dg', 'velocyto_hg', 'E9M2_Glial', 'E9-11F1_Glial', 'E9-11M2_Glial', 'E9-11F1_Gluta'}
data = 'EP'
SURFIX = '[dynamical]' if DYNAMICAL else ''
SURFIX += '[deep_velo]' if DEEPVELO else ''
