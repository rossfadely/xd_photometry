import os
import cPickle
import numpy as np

from xd import XDGMM
from utils import fetch_prepped_dr10data, fetch_prepped_s82data

# set data and model parameters
seed = 12345
epoch = 3
N = 60000
Nbatch = 30000
K = 32
uw = 0.5
Ncheck = 4
prev_iter = 0
n_iter = 128
Nstar = 16
data = 'dr12'
factor = 100.
features = ['psf_mag', 'model_colors', 'psf_minus_model']
filters = ['r', 'ug gr ri iz', 'ugriz']
message = 'pm_mc_pmm_r_all_all_v1'
total_iter = prev_iter + n_iter
model_parms_file = None
savefile = '../data/xdparms_%s_%d_%d_%d_%d_%s.hd5' % (data, N, K, total_iter,
                                                      Nstar, message)

# definitions for fixed means and aligned covs
fixed_mean_inds = [-5, -4, -3, -2, -1]
fixed_inds = fixed_mean_inds

# get the data
if data == 'dr12':
    m = N
    gname = 'dr12_30k_gals_rfadely.fit'
    sname = 'dr12_30k_stars_rfadely.fit'
    X, Xcov = fetch_prepped_dr10data(N, features=features, filters=filters,
                                     gname=gname, sname=sname)
    gname = 'dr12_30k_gals_2_rfadely.fit'
    sname = 'dr12_30k_stars_2_rfadely.fit'
    Xvalid, Xvalidcov = fetch_prepped_dr10data(N, features=features,
                                               filters=filters,
                                               gname=gname, sname=sname)
elif data == 's82':
    m = epoch
    X, Xcov = fetch_prepped_s82data(epoch, features=features, filters=filters)

# assign fixed means and aligned covs for stars
fixed_means = np.zeros((K, X.shape[1])) + np.inf
fixed_means[:Nstar, fixed_mean_inds] = np.zeros(len(fixed_mean_inds))
aligned_covs = [(i, fixed_inds) for i in range(Nstar)]

# regularize for singularities
w = np.ones(X.shape[1]) * np.inf
for i in range(X.shape[0]):
    w = np.minimum(w, np.diag(Xcov[i]))
w /= factor

model = XDGMM(X, Xcov, K, Nbatch, n_iter=n_iter, w=w, Nthreads=8,
              verbose=True, Xvalid=Xvalid, Xvalidcov=Xvalidcov,
              update_weight=uw, Ncheck=Ncheck,
              model_parms_file=model_parms_file, fixed_means=fixed_means,
              aligned_covs=aligned_covs,
              savefile=savefile,
              seed=seed)
