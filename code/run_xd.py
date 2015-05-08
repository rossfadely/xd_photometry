import os
import cPickle
import numpy as np
import pyfits as pf

from xd import XDGMM
from utils import fetch_prepped_data
ddir = os.environ['xddata']

# set data and model parameters
seed = 12345
N = 240000
Nval = 60000
batch_size = 4800
init_Nbatch = 6
nu = 0.75
K = 32
Nstar = 12
Nthreads = 12
Ncheck = 2
n_iter = 512
factor = 100.
features = ['psf_mag', 'model_colors', 'petror50', 'psf_minus_model']
filters = ['r', 'ug gr ri iz', 'ugriz', 'ugriz']

# parm file specifications.
#message = 'pm_mc_pmm_r_all_all_v1'
#model_parms_file = ddir + '/dr12_%d_%d_%d_%s.hdf5' % (N, K, Nstar, message)
model_parms_file = None
message = 'pm_mc_petro_pmm_r_all_all_all_v1'
savefile = ddir + '/dr12_%d_%d_%d_%s.hdf5' % (N, K, Nstar, message)

# training data files
message = message[:-3]
gdataname = 'dr12_120k_gals_rfadely.fit'
sdataname = 'dr12_120k_stars_rfadely.fit'
datafile = ddir + 'dr12_240_%s_design.fits' % message

# validation data files
gvalidname = 'dr12_30k_gals_valid_rfadely.fit'
svalidname = 'dr12_30k_stars_valid_rfadely.fit'

# definitions for fixed means and aligned covs
fixed_mean_inds = [-5, -4, -3, -2, -1]
fixed_inds = fixed_mean_inds

# create the data file if needed.
if not os.path.exists(datafile):
    print '\nCreating design matrix and covs.\n'
    X, Xcov = fetch_prepped_data(N, features=features, filters=filters,
                                 gname=gdataname, sname=sdataname,
                                 savefile=datafile)
else:
    f = pf.open(datafile)
    Xcov = f[1].data
    f.close()

# get valid data
Xvalid, Xvalidcov = fetch_prepped_data(Nval, features=features,
                                       filters=filters,
                                       gname=gvalidname, sname=svalidname)
ind = np.diag_indices(Xcov.shape[1])
w = Xcov[:, ind, ind].min(axis=0)[0] / 100.
del Xcov

# assign fixed means and aligned covs for stars
fixed_means = np.zeros((K, Xvalid.shape[1])) + np.inf
fixed_means[:Nstar, fixed_mean_inds] = np.zeros(len(fixed_mean_inds))
aligned_covs = [(i, fixed_inds) for i in range(Nstar)]

model = XDGMM(datafile, K, batch_size, n_iter=n_iter, Nthreads=Nthreads,
              verbose=True, Xvalid=Xvalid, Xvalidcov=Xvalidcov, w=w,
              Ncheck=Ncheck, init_Nbatch=init_Nbatch, nu=nu,
              model_parms_file=model_parms_file, fixed_means=fixed_means,
              aligned_covs=aligned_covs,
              savefile=savefile,
              seed=seed)
