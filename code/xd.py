"""
XD version originally by Jake Vanderplas/astroML: http://bit.ly/1FBM2Lg

Modifications by Ross Fadely:
- 2014-09-23: port and erase astroML dependencies.
- 2014-09-25: convert class to functions for easy multiprocessing,
              only Estep seems to be useful across mutliple threads.
- 2014-09-29: add simple covarince regularization to avoid sing. matricies.
- 2014-10-??: add functionality to compute posteriors for `true` values.
- 2014-11-03: add ability to fix means and align covariances.
- 2015-03-27: allow batches, validation set

Extreme deconvolution solver

This follows Bovy et al.
http://arxiv.org/pdf/0905.2979v2.pdf

Arbitrary mixing matrices R are not yet implemented: currently, this only
works with R = I.
"""

import multiprocessing
import numpy as np

from time import time
from sklearn.mixture import GMM
from utils import logsumexp, log_multivariate_gaussian
from utils import save_xd_parms, load_xd_parms, DataIterator
from utils import log_multivariate_gaussian_Nthreads
from gmm_wrapper import constrained_GMM
from interruptible_pool import InterruptiblePool

def XDGMM(datafile, n_components, batch_size, savefile=None,
          n_iter=100, tol=1E-5, Nthreads=1, R=None, Ncheck=32, valid_break=5, 
          init_Nbatch=None,
          Xvalid=None, Xvalidcov=None, update_weight=None,
          init_n_iter=10, w=None, model_parms_file=None, fixed_means=None,
          aligned_covs=None, seed=None, verbose=False):
    """
    Extreme Deconvolution

    Fit an extreme deconvolution (XD) model to the data

    Parameters
    ----------
    RF FILL IN!!

    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
    if seed is not None:
        np.random.seed(seed)

    # initialize data iterator and model.
    batch_itr = DataIterator(datafile, batch_size)
    model = xd_model(batch_itr.shape, n_components, n_iter, tol, w, Nthreads,
                     fixed_means, aligned_covs, verbose)

    if update_weight is None:
        update_weight = np.float(batch_size) / batch_itr.Ndata

    # if there is a given parm file, load them to model.
    if model_parms_file is not None:
        model.alpha, model.mu, model.V, _ = load_xd_parms(model_parms_file)

    # INITIALIZATION ---
    # initialize components via a few steps of GMM
    # this doesn't take into account errors, but is a fast first-guess
    t0 = time()
    X, Xcov = batch_itr.get_batch()
    if model.V is None:
        # get data to initialize
        if init_Nbatch is None:
            init_Nbatch = 1
        elif (init_Nbatch * batch_size > batch_itr.Ndata):
            assert False, 'initialization batch size too large'

        # add more data if requested for init.
        for i in range(init_Nbatch - 1):
            xt, xtc = batch_itr.get_batch()
            X = np.vstack((X, xt))
            Xcov = np.vstack((Xcov, xtc))
            del xt, xtc

        # launch (modified) scikit GMM
        if (fixed_means is None) & (aligned_covs is None):
            gmm = GMM(model.n_components, n_iter=init_n_iter,
                      covariance_type='full').fit(X)
        else:
            gmm = constrained_GMM(X, model.n_components, init_n_iter,
                                  fixed_means, aligned_covs)

        # assign parms
        model.mu = gmm.means_
        model.alpha = gmm.weights_
        model.V = gmm.covars_

    # initial likelihoods under GMM, or prev model
    model.train_logL = np.zeros(n_iter + 1)
    model.valid_logL = np.zeros(n_iter + 1)
    model.train_logL[0] = model.logLikelihood(X, Xcov)
    if Xvalid is not None:
        model.valid_logL[0] = model.logLikelihood(Xvalid, Xvalidcov)
        
    if model.verbose:
        print 'Initalization done in %.2g sec' % (time() - t0)
        print 'Initial Log Likelihood: ', model.train_logL[0]
        print 'Initial Valid Log Likelihood: ', model.valid_logL[0]

    # OPTIMIZATION --
    # Use EM in batches, stop on criteria based on validation set.
    tt = 0.0
    Nvalid_bad = 0
    for i in range(model.n_iter):
        t0 = time()

        # new batch
        X, Xcov = batch_itr.get_batch()

        # take a step and eval train likelihood
        model = _EMstep(model, X, Xcov, update_weight)
        model.train_logL[i + 1] = model.logLikelihood(X, Xcov)

        # only update validation likelihood every so often, can cost 
        # as much or more than EM update depending on Nvalid.
        if (i % Ncheck == 0):
            model.valid_logL[i + 1] = model.logLikelihood(Xvalid, Xvalidcov)

        t1 = time()
        tt += t1 - t0
        if model.verbose:
            tup = (i + 1, model.train_logL[i + 1], model.valid_logL[i + 1])
            print "%i: log(L) = %.5g, log(Lvalid) = %.5g" % tup
            print "iter:%.2g sec, total:%.2g sec" % ((t1 - t0), tt)

        # save parms and stop, if needed.
        if (i % Ncheck == 0):
            if (model.valid_logL[i + 1] > model.valid_logL[i]):
                save_xd_parms(savefile, model.alpha, model.mu, model.V,
                              model.valid_logL)
            elif Xvalid is None:
                save_xd_parms(savefile, model.alpha, model.mu, model.V,
                              model.train_logL)

            it = np.minimum(n_iter - 1, 3) # run for three iters min.
            if model.valid_logL[i + 1] < np.max(model.valid_logL[it:]):
                Nvalid_bad += 1
                if Nvalid_bad == valid_break:
                    print 'Bad valid, stopping', i, Nvalid_bad
                    del X, Xcov, batch_itr # paranoid cleanup.
                    break

    return model

def _EMstep(model, X, Xcov, update_weight):
    """
    Perform the E-step (eq 16 of Bovy et al)
    """
    X = X[:, np.newaxis, :]
    Xcov = Xcov[:, np.newaxis, :, :]

    w_m = X - model.mu
    T = Xcov + model.V

    # Estep
    if model.Nthreads > 1:
        q, b, B = _Estep_multi(model, T, w_m, X)
    elif model.Nthreads == 1:
        q, b, B = _Estep((T, w_m, X, model))
    else:
        assert False, 'Number of threads must be greater than 0.'

    # M step
    model = _Mstep(model, q, b, B, update_weight)

    return model

def _Estep((T, w_m, X, model)):
    """
    Compute the Estep for the given data chunk and current model
    """

    # compute inverse of each covariance matrix T
    Tshape = T.shape
    T = T.reshape([X.shape[0] * model.n_components,
                   model.n_features, model.n_features])
    Tinv = np.array([np.linalg.inv(T[i])
                     for i in range(T.shape[0])]).reshape(Tshape)
    T = T.reshape(Tshape)

    # evaluate each mixture at each point
    N = np.exp(log_multivariate_gaussian(X, model.mu, T, Vinv=Tinv))

    # q
    q = (N * model.alpha) / np.dot(N, model.alpha)[:, None]

    # b
    tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
    b = model.mu + np.sum(model.V * tmp[:, :, np.newaxis, :], -1)

    # B
    tt = time()
    tmp = np.sum(Tinv[:, :, :, :, np.newaxis]
                 * model.V[:, np.newaxis, :, :], -2)
    B = model.V - np.sum(model.V[:, :, :, np.newaxis]
                         * tmp[:, :, np.newaxis, :, :], -2)

    return (q, b, B)

def _Estep_multi(model, T, w_m, X):
    """
    Use multiple processes to compute Estep.
    """
    pool = InterruptiblePool(model.Nthreads)
    mapfn = pool.map
    Nchunk = np.ceil(1. / model.Nthreads * X.shape[0]).astype(np.int)

    arglist = [None] * model.Nthreads
    for i in range(model.Nthreads):
        s = i * Nchunk
        e = s + Nchunk
        arglist[i] = (T[s:e], w_m[s:e], X[s:e], model)

    results = list(mapfn(_Estep, [args for args in arglist]))
    q, b, B = results[0]
    for i in range(1, model.Nthreads):
        q = np.vstack((q, results[i][0]))
        b = np.vstack((b, results[i][1]))
        B = np.vstack((B, results[i][2]))

    pool.close()
    pool.terminate()
    pool.join()
    return q, b, B

def _Mstep(model, q, b, B, update_weight):
    """
    M-step: compute alpha, mu, V, update to model
    """
    Nbatch = q.shape[0]
    qj = q.sum(0)
    alpha = qj / Nbatch

    # prevent no component from having too low a weight
    ind = alpha < model.min_alpha
    alpha[ind] = model.min_alpha
    alpha /= alpha.sum()
    qj = alpha * Nbatch

    # update alpha
    d_alpha = alpha - model.alpha
    model.alpha = d_alpha * update_weight + model.alpha

    # update mu
    d_mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis] - model.mu
    model.mu = d_mu * update_weight + model.mu
    if model.fixed_means is not None:
        ind = model.fixed_means < np.inf
        model.mu[ind] = model.fixed_means[ind]

    # update V
    m_b = model.mu - b
    tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
    tmp += B
    tmp *= q[:, :, np.newaxis, np.newaxis]
    d_V = (tmp.sum(0) + model.w[np.newaxis, :, :]) / \
          (qj[:, np.newaxis, np.newaxis] + 1) - model.V
    model.V = d_V * update_weight + model.V

   # axis align covs if desired
    if model.aligned_covs is not None:
        for info in model.aligned_covs:
            c, inds = zip(info)
            s = model.V[c, inds, inds]
            model.V[c, inds, :] = 0.0
            model.V[c, :, inds] = 0.0
            model.V[c, inds, inds] = s

    return model


class xd_model(object):
    """
    Class to store all things pertinent to the XD model. 
    """
    def __init__(self, xshape, n_components, n_iter, tol, w, Nthreads,
                 fixed_means, aligned_covs, verbose, min_alpha=1.e-100):
        self.n_samples = xshape[0]
        self.n_features = xshape[1]
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.Nthreads = Nthreads
        self.fixed_means = fixed_means
        self.aligned_covs = aligned_covs
        self.verbose = verbose
        self.min_alpha = min_alpha

        self.V = None
        self.mu = None
        self.alpha = None

        # construct simple cov regularization term.  
        # regularize only along the diagonal, and same for each component.
        if type(w) == float:
            w = np.eye(self.n_features) * w
        elif type(w) == np.ndarray:
            if w.size == self.n_features:
                w = np.diag(w)
        else:
            w = np.diag(np.zeros(self.n_features))
        self.w = w

    def logprob_a(self, X, Xcov):
        """
        Evaluate the probability for a set of points

        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xcov: array_like
            Covariance of input data.  shape = (n_samples, n_features,
            n_features)

        Returns
        -------
        p: ndarray
            Probabilities.  shape = (n_samples,)
        """
        X = np.asarray(X)
        Xcov = np.asarray(Xcov)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xcov.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]
        Xcov = Xcov[:, np.newaxis, :, :]
        T = Xcov + self.V

        if self.Nthreads == 1:
            return log_multivariate_gaussian(X, self.mu, T)
        else:
            return log_multivariate_gaussian_Nthreads(X, self.mu, T,
                                                      self.Nthreads)

    def logLikelihood(self, X, Xcov):
        """
        Compute the log-likelihood of data given the model

        Parameters
        ----------
        X: array_like
            data, shape = (n_samples, n_features)
        Xcov: array_like
            covariances, shape = (n_samples, n_features, n_features)

        Returns
        -------
        logL : float
            log-likelihood
        """
        return np.mean(logsumexp(self.logprob_a(X, Xcov), -1))

    def sample(self, alpha, mu, V, size=1):
        shape = tuple(np.atleast_1d(size)) + (mu.shape[1],)
        npts = np.prod(size)

        alpha_cs = np.cumsum(alpha)
        r = np.atleast_1d(np.random.random(size))
        r.sort()

        ind = r.searchsorted(alpha_cs)
        ind = np.concatenate(([0], ind))
        if ind[-1] != size:
            ind[-1] = size

        draw = np.vstack([np.random.multivariate_normal(mu[i],
                                                        V[i],
                                                        (ind[i + 1] - ind[i],))
                          for i in range(len(alpha))])

        return draw.reshape(shape)

    def posterior(self, X, Xcov):
        """
        Return the posterior mean and covariance given the xd model.
        """
        draw = None
        post_alpha = np.zeros((X.shape[0], self.alpha.size))
        post_mu = np.zeros((X.shape[0], self.alpha.size, X.shape[1]))
        post_V = np.zeros((Xcov.shape[0], self.alpha.size,
                           Xcov.shape[1], Xcov.shape[2]))

        iV = np.zeros_like(self.V)
        for i in range(self.n_components):
            iV[i] = np.linalg.inv(self.V[i])

        for i in range(Xcov.shape[0]):
            if (model.verbose & i % 1000):
                print 'Posterior Calculation for datum %d' % i
            Xicov = np.linalg.inv(Xcov[i])
            for j in range(self.n_components):
                dlt = X[i] - self.mu[j]
                convV = self.V[j] + Xcov[i]
                iconvV = np.linalg.inv(convV) 
                post_V[i, j] = np.linalg.inv(iV[j] + Xicov)
                post_mu[i, j] = np.dot(iV[j], self.mu[j])
                post_mu[i, j] += np.dot(Xicov, X[i])
                post_mu[i, j] = np.dot(post_V[i, j], post_mu[i, j])
                post_alpha[i, j] = np.dot(np.dot(dlt.T, iconvV), dlt)
                post_alpha[i, j] = np.exp(-0.5 * post_alpha[i, j])
                post_alpha[i, j] /= np.sqrt(2 * np.pi * np.linalg.det(convV))
            post_alpha[i] /= post_alpha[i].sum()

        return post_alpha, post_mu, post_V
