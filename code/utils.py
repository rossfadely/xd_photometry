# 
# Various routines, mostly for handing data.
#
# Author - Ross Fadely (unless otherwise noted).
#
import os
import h5py
import psutil
import numpy as np
import pyfits as pf

from interruptible_pool import InterruptiblePool

def mem():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    print 'Memory usage is %0.1f GB' % (mem / 1000.)

def log_multivariate_gaussian_Nthreads(x, mu, V, xcov, Nthreads=1):
    """
    Use multiprocessing to calculate log likelihoods.
    """
    n_samples = x.shape[0]
    pool = pool = InterruptiblePool(Nthreads)
    mapfn = pool.map
    Nchunk = np.ceil(1. / Nthreads * n_samples).astype(np.int)

    arglist = [None] * Nthreads
    for i in range(Nthreads):
        s = i * Nchunk
        e = s + Nchunk
        arglist[i] = (x[s:e], mu, V, xcov[s:e])

    result = list(mapfn(lmg, [args for args in arglist]))

    logls = result[0]
    for i in range(1, Nthreads):
       logls = np.vstack((logls, result[i]))
       
    pool.close()
    pool.terminate()
    pool.join()
    return logls

def lmg(args):
    from log_multi_gauss import logmultigauss
    return logmultigauss(*args)
    #x, m, v, xc = args
    #X = x[:, np.newaxis, :]
    #Xcov = xc[:, np.newaxis, :, :]
    #print Xcov.shape, X.shape, v.shape
    #T = Xcov + v
    #print T.shape
    #return log_multivariate_gaussian(X, m, T)

def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1):
    """
    Swiped from astroML:
    https://github.com/astroML/astroML/blob/master/astroML/utils.py

    Evaluate a multivariate gaussian N(x|mu, V)

    This allows for multiple evaluations at once, using array broadcasting

    Parameters
    ----------
    x: array_like
        points, shape[-1] = n_features

    mu: array_like
        centers, shape[-1] = n_features

    V: array_like
        covariances, shape[-2:] = (n_features, n_features)

    Vinv: array_like or None
        pre-computed inverses of V: should have the same shape as V

    method: integer, optional
        method = 0: use cholesky decompositions of V
        method = 1: use explicit inverse of V

    Returns
    -------
    values: ndarray
    shape = broadcast(x.shape[:-1], mu.shape[:-1], V.shape[:-2])

    Examples
    --------

    >>> x = [1, 2]
    >>> mu = [0, 0]
    >>> V = [[2, 1], [1, 2]]
    >>> log_multivariate_gaussian(x, mu, V)
    -3.3871832107434003
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    if Vinv is not None:
        assert Vinv.shape == Vshape
        method = 1

    if method == 0:
        Vchol = np.array([np.linalg.cholesky(V[i])
                          for i in range(V.shape[0])])

        # we may be more efficient by using scipy.np.linalg.solve_triangular
        # with each cholesky decomposition
        VcholI = np.array([np.linalg.inv(Vchol[i])
                          for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

    elif method == 1:
        if Vinv is None:
            Vinv = np.array([np.linalg.inv(V[i])
                             for i in range(V.shape[0])]).reshape(Vshape)
        else:
            assert Vinv.shape == Vshape

        logdet = np.log(np.array([np.linalg.det(V[i])
                                  for i in range(V.shape[0])]))
        logdet = logdet.reshape(Vshape[:-2])

        xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
        xVIx = np.sum(xVI * x_mu, -1)

    else:
        raise ValueError("unrecognized method %s" % method)

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)

def logsumexp(arr, axis=None):
    """
    Swiped from astroML:
    https://github.com/astroML/astroML/blob/master/astroML/utils.py
    
    Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    # if axis is specified, roll axis to 0 so that broadcasting works below
    if axis is not None:
        arr = np.rollaxis(arr, axis)
        axis = 0

    # Use the max to normalize, as with the log this is what accumulates
    # the fewest errors
    vmax = arr.max(axis=axis)
    out = np.log(np.sum(np.exp(arr - vmax), axis=axis))
    out += vmax

    return out

def fetch_mixed_epoch(epoch, gal_frac=0.5, shuffle=True, seed=12345):
    """
    Make a combined sample of stars and galaxies for a given epoch.  Return
    single epoch and coadd data.
    """
    # fetch
    ss, sc = fetch_epoch(epoch, 'stars')
    gs, gc = fetch_epoch(epoch, 'gals')

    # size
    N = np.minimum(ss.field(0).size, gs.field(0).size)
    Ngal = np.round(gal_frac * N)
    Nstar = N - Ngal

    # shuffle:
    if shuffle:
        np.random.seed(seed)
        ind = np.random.permutation(N).astype(np.int)
    else:
        ind = np.arange(N, dtype=np.int)

    # cut
    ss, sc = ss[:Nstar], sc[:Nstar]
    gs, gc = gs[:Ngal], gc[:Ngal]

    # build
    os = {}
    oc = {}
    for k in ss._coldefs.names:
        os[k] = np.append(ss.field(k), gs.field(k))[ind]
        if k != 'coadd_objid':
            oc[k] = np.append(sc.field(k), gc.field(k))[ind]

    return os, oc

def fetch_epoch(epoch, kind, verbose=False):
    """
    Return the single epoch and the matched coadded data.
    """
    assert kind in ['stars', 'gals']
    ddir = os.environ['xddata']

    # single epoch
    f = pf.open(ddir + 's82single_%s_%d.fits' % (kind, epoch))
    s = f[1].data
    f.close()
    N = s.field(0).size

    try:
        f = pf.open(ddir + 's82coadd_%s_%d.fits' % (kind, epoch))
        c = f[1].data
        f.close()

    except:
        print 'Matched fits for coadd doesn\'t exist, building...'

        # master
        f = pf.open(ddir + 's82coadd30k_%s_rfadely.fit' % kind)
        c = f[1].data
        f.close()

        # find the corresponding coadds
        inds = np.zeros(N, dtype=np.int)
        ind = 0
        for i in range(N):
            if verbose:
                if i % 200 == 0:
                    print 'searching', i
            coadd_objid = s.field('coadd_objid')[i]
            search = True
            while search:
                if c.field('objid')[ind] == coadd_objid:
                    inds[i] = ind
                    search = False
                ind += 1

        c = c[inds]
        if False:
            # paranoid check
            for i in range(N):
                st = '%d %d' % (c[i].field('objid'), s[i].field('coadd_objid'))
                assert c[i].field('objid') == s[i].field('coadd_objid'), st

        dt = {'E':np.float32, 'K':np.int64, 'D':np.float64, 'I':np.int16,
              'K':np.int64}
        cols = []
        for i in range(len(s[0]) - 1):
            n = s._coldefs.names[i]
            f = s._coldefs.formats[i]
            cols.append(pf.Column(name=n, format=f,
                                  array=c.field(n).astype(dt[f])))

        tbhdu = pf.new_table(pf.ColDefs(cols))
        tbhdu.writeto(ddir + 's82coadd_%s_%d.fits' % (kind, epoch),
                      clobber=True)
        c = tbhdu.data
    
    return s, c

def fetch_matched_s82data(epoch, fgal=0.5, features=['psf_mag',
                                                     'model_colors',
                                                     'psf_minus_model'],
                          filters=['r', 'ur gr ri rz', 'r'], use_single=True):
    """
    Construct data matrix and cov.
    """
    single, coadd = fetch_mixed_epoch(epoch, fgal)

    if use_single:
        d = single
    else:
        d = coadd
    return prep_data(d, features, filters)

def fetch_single_prepped_data(filename, features=['psf_mag', 'model_colors',
                                                  'psf_minus_model'],
                              filters=['r', 'ur gr ri rz', 'r'],
                              seed=1234):
    ddir = os.environ['xddata']
    f = pf.open(ddir + filename)
    d = f[1].data
    f.close()
    return prep_data(d, features, filters)

def fetch_prepped_data(N, fgal=0.5, features=['psf_mag', 'model_colors',
                                              'psf_minus_model'],
                       filters=['r', 'ur gr ri rz', 'r'],
                       seed=1234, gname=None, sname=None, savefile=None,
                       return_labels=False):
    """
    Prepare SDSS data to run XD.
    """
    if gname is None:
        gname = 'dr12_30k_gals_rfadely.fit'
        sname = 'dr12_30k_stars_rfadely.fit'
    np.random.seed(seed)
    ddir = os.environ['xddata']
    f = pf.open(ddir + sname)
    d = f[1].data
    f.close()
    Xstar, Xstarcov = prep_data(d, features, filters)
    f = pf.open(ddir + gname)
    d = f[1].data
    f.close()
    Xgal, Xgalcov = prep_data(d, features, filters)
    
    Nmax = N * fgal
    assert Xgal.shape[0] >= Nmax, 'Not enough galaxy data for request'
    Nmax = N - Nmax
    assert Xstar.shape[0] >= Nmax, 'Not enough star data for request'

    Ngal = np.round(N * fgal).astype(np.int)
    Nstar = N - Ngal
    ind = np.random.permutation(Xgal.shape[0])[:Ngal]
    Xgal = Xgal[ind]
    Xgalcov = Xgalcov[ind]
    gl = np.zeros(Ngal)
    ind = np.random.permutation(Xstar.shape[0])[:Nstar]
    Xstar = Xstar[ind]
    Xstarcov = Xstarcov[ind]
    sl = np.ones(Nstar)
    
    X = np.vstack((Xgal, Xstar))
    Xcov = np.vstack((Xgalcov, Xstarcov))
    labels = np.append(gl, sl)
    ind = np.random.permutation(X.shape[0])
    X = X[ind]
    Xcov = Xcov[ind]
    labels = labels[ind]

    if savefile is not None:
        prihdu = pf.PrimaryHDU(X)
        sechdu = pf.ImageHDU(Xcov)
        hdulist = pf.HDUList([prihdu, sechdu])
        hdulist.writeto(savefile, clobber=True)

    if return_labels:
        return X, Xcov, labels
    else:
        return X, Xcov

def make_W_matrix(features, filters, odim):
    """
    Construct the mixing matrix for the set of features.  Assumed here is that 
    the first five elements of the unmixed matrix are psf mags, the next 
    five are model mags.
    """
    ref = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4}
    W = np.zeros((1, odim))
    idx = 0
    for i, feature in enumerate(features):
        # begin spaghetti code
        if 'psf' in feature:
            ind = 0
        elif 'model' in feature:
            ind = 5
        elif 'petro' in feature:
            ind = 10
        elif 'fwhm' in feature:
            ind = 15
        else:
            assert 0, 'Feature not implemented'

        if ('mag' in feature) or ('petro' in feature):
            for f in filters[i]:
                W = np.vstack((W, np.zeros((1, odim))))
                W[idx, ind + ref[f]] = 1.
                idx += 1

        if ('fwhm' in feature):
            W = np.vstack((W, np.zeros((1, odim))))
            W[idx, ind] = 1.
            idx += 1

        if 'colors' in feature:
            filts = filters[i].split()
            for f in filts:
                W = np.vstack((W, np.zeros((1, odim))))
                W[idx, ind + ref[f[0]]] = 1.
                W[idx, ind + ref[f[1]]] = -1.
                idx += 1
        
        if 'minus' in feature:
            for f in filters[i]:
                W = np.vstack((W, np.zeros((1, odim))))
                W[idx, ref[f]] = 1.
                W[idx, 5 + ref[f]] = -1.
                idx += 1

    return W[:-1]

def prep_data(d, features, filters=None, max_err=1000., s=0.396,
              acs_fwhm_err=0.01):
    """
    Return the prepared data.
    """
    if filters is None:
        filters = ['ugriz' for i in range(len(features))]

    psfmags = np.vstack([d['psfmag_' + f] -
                         d['extinction_' + f] for f in 'ugriz']).T
    psfmagerrs = np.vstack([d['psfmagerr_' + f] for f in 'ugriz']).T
    modelmags = np.vstack([d['modelmag_' + f] -
                           d['extinction_' + f]for f in 'ugriz']).T
    modelmagerrs = np.vstack([d['modelmagerr_' + f] for f in 'ugriz']).T

    try:
        seeing = np.vstack([d['mrrcc_' + f] for f in 'ugriz']).T
        ind = seeing < 0.0
        seeing = 2. * s * np.sqrt(np.log(2.) * np.abs(seeing))
        petror50s = np.vstack([d['petror50_' + f] for f in 'ugriz']).T
        petror50errs = np.vstack([d['petror50err_' + f] for f in 'ugriz']).T
        ind = ind | (petror50s < 0.) | (np.abs(petror50errs) > max_err)
        seeing[ind] = np.median(seeing[ind != True])
        petror50s[ind] = np.median(petror50s[ind != True])
        petror50errs[ind] = max_err
        petror50s /= seeing
        petror50errs /= seeing
    except:
        petror50s = 0.5 * np.ones_like(psfmags)
        petror50errs = max_err * np.ones_like(psfmags)

    try:
        fwhms = np.vstack([d['acs_fwhm']]).T
        fwhmerrs = acs_fwhm_err * np.ones_like(fwhms)
    except:
        fwhms = 0.5 * np.vstack([np.ones(psfmags.shape[0])]).T
        fwhmerrs = acs_fwhm_err * np.ones_like(fwhms)

    X = np.hstack((psfmags, modelmags, petror50s, fwhms))
    Xerr = np.hstack((psfmagerrs, modelmagerrs, petror50errs, fwhmerrs))

    W = make_W_matrix(features, filters, X.shape[1])

    X = np.dot(X, W.T)
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr ** 2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    return X, Xcov

def fetch_glob_data(name, features, filters):
    """
    Return the data and cov matrix for SDSS DR10 data for the specified
    globular cluster.
    """
    assert name.lower() in ['m3', 'm5', 'm15']
    from glob import glob
    files = glob(os.environ['xddata'] + name.lower() + '*fits')

    f = pf.open(files[0])
    d = f[1].data
    d = d[(d['clean'] == 1) & (d['type'] == 6)]
    X, Xcov = prep_data(d, features, filters)
    f.close()

    if len(files) > 1:
        for i in range(1, len(files)):
            f = pf.open(files[i])
            d =f[1].data
            d =d[(d['clean'] == 1) & (d['type'] == 6)]
            tx, txc = prep_data(d, features, filters)
            f.close()
            X = np.vstack((X, tx))
            Xcov = np.vstack((Xcov, txc))

    return X, Xcov

def save_xd_parms(filename, a, m, v, trn_logl, vld_logl):
    """
    Save the XDGMM params in a fits table.
    """
    if filename is None:
        return
    f = h5py.File(filename, 'w')
    f.create_dataset('alpha', data=a)
    f.create_dataset('mu', data=m)
    f.create_dataset('V', data=v)
    f.create_dataset('train_logl', data=trn_logl)
    f.create_dataset('valid_logl', data=vld_logl)
    f.close()

def load_xd_parms(filename):
    """
    Return an xd model class instance with the params in the hdf file.
    """
    f = h5py.File(filename,'r')
    alpha = f['alpha'][:]
    mu = f['mu'][:]
    V = f['V'][:]
    train_logl = f['train_logl'][:]
    valid_logl = f['valid_logl'][:]
    f.close()
    return alpha, mu, V, train_logl, valid_logl

class DataIterator(object):
    """
    Batch iteration of data, reading batch by batch from disk.
    """
    def __init__(self, datafile, batch_size):
        self.datafile = datafile

        f = pf.open(datafile)
        self.Xmmap = f[0].data
        self.Xcovmmap = f[1].data
        self.Ndata = len(self.Xmmap)
        self.Ndim = len(self.Xmmap[0])
        self.shape = (self.Ndata, self.Ndim)
        f.close()

        self.batch_size = batch_size
        self.start = 0
        self.end = min(self.Ndata, batch_size)

    def get_batch(self):
        f = pf.open(self.datafile)
        X = f[0].data[self.start:self.end]
        Xcov = f[1].data[self.start:self.end]
        f.close()

        self.iterate()
        return X, Xcov

    def iterate(self):
        self.start += self.batch_size
        self.end += self.batch_size
        if self.start >= self.Ndata:
            self.start = 0
            self.end = self.batch_size

if __name__ == '__main__':
    #features = ['psf_mag', 'model_colors', 'psf_minus_model', 'acs_fwhm']
    #filters = ['r', 'ug gr ri iz', 'ugriz', 'i']
    #features = ['psf_mag', 'model_colors', 'acs_fwhm']
    #filters = ['r', 'ug gr ri iz', 'i']
    #features = ['psf_mag', 'model_colors', 'acs_fwhm']
    #filters = ['r', 'ur', 'i']
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    features = ['psf_mag', 'model_colors']
    filters = ['r', 'gr']

    ddir = os.environ['xddata']
    f = 'dr12_cosmos.fits'
    f = pf.open(ddir + f)
    d = f[1].data
    f.close()
    X, Xcov = prep_data(d, features, filters)

    """
    f = 'dr12_240_pm_mc_pmm_r_all_all_design.fits'
    f = pf.open(ddir + f)
    print f[1].data[0]
    f.close()
    gdataname = 'dr12_120k_gals_rfadely.fit'
    sdataname = 'dr12_120k_stars_rfadely.fit'
    X, Xcov = fetch_prepped_data(240000, features=features, filters=filters,
                                 gname=gdataname, sname=sdataname)

    """
