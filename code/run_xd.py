import os
import cPickle
import numpy as np

from xd import XDGMM
from utils import fetch_prepped_dr12data, fetch_prepped_s82data
ddir = os.environ['xddata']

# set data and model parameters
seed = 12345
epoch = 3
N = 60000
batch_size = 30000
K = 32
uw = 0.5
Ncheck = 4
n_iter = 1
Nstar = 16
data = 'dr12'
factor = 100.
features = ['psf_mag', 'model_colors', 'psf_minus_model']
filters = ['r', 'ug gr ri iz', 'ugriz']

# parm file specifications.
total_iter = n_iter
message = 'pm_mc_pmm_r_all_all_v1'
model_parms_file = ddir + '/xdparms_%s_%d_%d_%d_%d_%s.hd5' % (data, N, K,
                                                              total_iter,
                                                              Nstar, message)
message = 'pm_mc_pmm_r_all_all_v1'
savefile = ddir + '/xdparms_%s_%d_%d_%d_%d_%s.hd5' % (data, N, K, total_iter,
                                                      Nstar, message)

# training data files
gdataname = 'dr12_30k_gals_rfadely.fit'
sdataname = 'dr12_30k_stars_rfadely.fit'
datafile = ddir + 'dr12_60k_design.fits'

# validation data files
gvalidname = 'dr12_30k_gals_2_rfadely.fit'
svalidname = 'dr12_30k_stars_2_rfadely.fit'

# definitions for fixed means and aligned covs
fixed_mean_inds = [-5, -4, -3, -2, -1]
fixed_inds = fixed_mean_inds

# create the data file if needed.
if not os.path.exists(datafile):
    X, Xcov = fetch_prepped_dr12data(N, features=features, filters=filters,
                                     gname=gdataname, sname=sdataname,
                                     savefile=datafile)

# get valid data
Xvalid, Xvalidcov = fetch_prepped_dr12data(N, features=features,
                                           filters=filters,
                                           gname=gvalidname, sname=svalidname)

# assign fixed means and aligned covs for stars
fixed_means = np.zeros((K, Xvalid.shape[1])) + np.inf
fixed_means[:Nstar, fixed_mean_inds] = np.zeros(len(fixed_mean_inds))
aligned_covs = [(i, fixed_inds) for i in range(Nstar)]

model = XDGMM(datafile, K, batch_size, n_iter=n_iter, Nthreads=8,
              verbose=True, Xvalid=Xvalid, Xvalidcov=Xvalidcov,
              update_weight=uw, Ncheck=Ncheck,
              model_parms_file=model_parms_file, fixed_means=fixed_means,
              aligned_covs=aligned_covs,
              savefile=savefile,
              seed=seed)
