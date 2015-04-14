#
# Plotting routines (mostly for paper).  Functions for paper figures are 
# labelled as fig_XX
#
# Author - Ross Fadely
#
from matplotlib import use; use('Agg')
from utils import fetch_glob_data, fetch_matched_s82data
from utils import fetch_prepped_data, load_xd_parms
from utils import log_multivariate_gaussian_Nthreads, logsumexp
from astroML.plotting.tools import draw_ellipse
from triangle import hist2d, error_ellipse

import os
import cPickle
import numpy as np
import matplotlib.pyplot as pl

def fig_1():
    """
    Produce figure 1 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig1.png'
    contours_and_data(epoch, model, features, filters, figname, idx=-3,
                      data=data)

def fig_2():
    """
    Produce figure 2 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig2.png'
    posteriors_plot(model, features, filters, figname, idx=-3)

def fig_3():
    """
    Produce figure 3 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig3.png'
    error_rates(epoch, model, features, filters, figname, idx=-3, N=10000)

def fig_4():
    """
    Produce figure 4 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig4.png'
    xx_plot(epoch, model, features, filters, figname)

def fig_5():
    """
    Produce figure 5 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig5.png'
    gn = 'm3'
    glob_cmd(model, gn, features, filters, figname)

def fig_6():
    """
    Produce figure 5 of the paper.
    """
    epoch = 3
    N = 60000
    Nr = N
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    factor = 100.
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    model = 'xdmodel_%s_%d_%d_%d_%d_%s.pkl' % (data, Nr, K, n_iter, Nstar,
                                               message)
    model = os.environ['xddata'] + model
    figname = os.environ['xdplots'] + 'fig6.png'
    psfminusmodel_plot(epoch, model, features, filters, figname, idx=-3)

def bininator(magbins, dlt, mags, err):
    """
    Bin up errors and return medians.
    """
    meds = np.zeros_like(magbins)
    for i in range(len(magbins)):
        ind = (mags > magbins[i] - dlt) & (mags <= magbins[i] + dlt)
        ind = ind & (err < 10.)
        meds[i] = np.median(err[ind])
    return meds

def error_rates(epoch, model, features, filters, figname, fgal=0.5, 
                idx=-1, N=10000):
    """
    Plot the median error rates for single epoch, dr10, coadd, and
    posteriors.
    """
    Xsingle, Xsinglecov = fetch_prepped_s82data(epoch, fgal, features, filters)
    Xcoadd, Xcoaddcov = fetch_prepped_s82data(epoch, fgal, features,
                                              filters, use_single=False)
    Xdr10, Xdr10cov = fetch_prepped_dr10data(60000, fgal, features, filters)

    Xdr10 = Xdr10[:N]
    Xdr10cov = Xdr10cov[:N]
    Xsingle = Xsingle[:N]
    Xsinglecov = Xsinglecov[:N]
    Xcoadd = Xcoadd[:N]
    Xcoaddcov = Xcoaddcov[:N]

    # unpickle the XD model
    if type(model) == str:
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    a1, m1, v1 = model.posterior(Xsingle, Xsinglecov)
    a2, m2, v2 = model.posterior(Xdr10, Xdr10cov)

    dr10_med = np.zeros_like(Xdr10)
    single_med = np.zeros_like(Xsingle)
    dr10_sig = np.zeros_like(Xdr10)
    single_sig = np.zeros_like(Xsingle)
    for i in range(N):
        samp = model.sample(a1[i], m1[i], v1[i], size=1000)
        single_med[i] = np.median(samp, axis=0)
        single_sig[i] = np.std(samp, axis=0)
        samp = model.sample(a2[i], m2[i], v2[i], size=1000)
        dr10_med[i] = np.median(samp, axis=0)
        dr10_sig[i] = np.std(samp, axis=0)

    fs = 5
    dlt = 0.2
    fac = 2
    lsize = 20
    mags = np.linspace(18, 22, 4. / dlt)
    ind = [0, 1, 2, idx, 3, 4]
    ylab = ['psfmag $r$ error', 'modelmag $u-g$ error',
            'modelmag $g-r$ error', 'psfmag - modelmag $r$ error',
            'modelmag $r-i$ error', 'modelmag $i-z$ error']
    xlab = 'psfmag $r$'
    xticks = np.array(['%0.0f' % v for v in np.linspace(18, 22, 9)])
    xticks[range(1, 8, 2)] = ''
    f = pl.figure(figsize=(3 * fs, 2 * fs))
    pl.subplots_adjust(wspace=0.3)
    for i in range(len(ind)):
        dr10err = bininator(mags, dlt, Xdr10[:, 0],
                            np.sqrt(Xdr10cov[:, ind[i]][:, ind[i]]))
        singleerr = bininator(mags, dlt, Xsingle[:, 0],
                            np.sqrt(Xsinglecov[:, ind[i]][:, ind[i]]))
        coadderr = bininator(mags, dlt, Xcoadd[:, 0],
                            np.sqrt(Xcoaddcov[:, ind[i]][:, ind[i]]))
        dr10_posterr = bininator(mags, dlt, dr10_med[:, 0],
                                 dr10_sig[:, ind[i]])
        single_posterr = bininator(mags, dlt, single_med[:, 0],
                                   single_sig[:, ind[i]])
        ax = pl.subplot(2, 3, i + 1)
        pl.plot(mags, coadderr, 'k', lw=2, label='coadd')
        pl.plot(mags, singleerr, 'k', ls=':', lw=2, label='single epoch')
        pl.plot(mags, single_posterr, 'k--', lw=2,
                label='XD post. single epoch')
        pl.plot(mags, dr10err, 'r', ls=':', lw=2, label='DR10')
        pl.plot(mags, dr10_posterr, 'r--', lw=2, label='XD post. DR10')
        pl.xlabel(xlab, fontsize=lsize)
        pl.ylabel(ylab[i], fontsize=lsize)
        pl.xlim(18, 22)
        pl.ylim(-fac * coadderr[0], singleerr.max())
        ax.set_xticklabels(xticks)
        if i == 1:
            pl.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5,
                      prop={'size':15})
    f.savefig(figname, bbox_inches='tight')

def contours_and_data(epoch, model, features, filters, figname, fgal=0.5,
                      idx=-1, data='s82', N=60000):
    """
    Plot the data and contours for objects called star/galaxy.
    """
    if data == 's82':
        # fetch Stripe 82 data
        X, Xcov = fetch_prepped_s82data(epoch, fgal, features, filters)
        Xcoadd, Xcoaddcov = fetch_prepped_s82data(epoch, fgal, features,
                                                  filters, use_single=False)
        sind = np.abs(Xcoadd[:, idx]) < 0.03
        gind = np.abs(Xcoadd[:, idx]) > 0.03

    else:
        # fetch DR10 data
        X, Xcov = fetch_prepped_dr10data(N, fgal, features, filters)
        sind = np.abs(X[:, idx]) < 0.145
        gind = np.abs(X[:, idx]) > 0.145

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    fs = 5
    ms = 1
    lsize = 20
    idx = [[0, -1], [2, 3], [3, 4]]
    xlim = [(18., 22), (-0.5, 2.5), (-0.5, 2)]
    ylim = [(-0.1, 0.5), (-0.5, 2.5), (-0.5, 1.5)]
    xlab = ['psfmag $r$', 'modelmag $g-r$', 'modelmag $r-i$']
    ylab = ['psfmag - modelmag $r$', 'modelmag $r-i$', 'modelmag $i-z$']

    f = pl.figure(figsize=(3 * fs, 3 * fs))
    Nstar = len(np.where(model.fixed_means[:, idx] != np.inf)[0])
    pl.subplots_adjust(wspace=0.3)
    for i in range(1, 10):
        k = (i - 1) % 3
        if i < 4:
            ind = np.arange(X.shape[0], dtype=np.int)
            rng = range(model.n_components)
        elif 3 < i < 7:
            ind = sind
            rng = range(Nstar)
        else:
            ind = gind
            rng = range(Nstar, model.n_components)
        ax = pl.subplot(3, 3, i)
        for j in rng:
            if model.alpha[j] > 1.e-3:
                draw_ellipse(model.mu[j, idx[k]],
                             model.V[j, idx[k]][:, idx[k]],
                             scales=[2], ec='k', fc='gray', alpha=0.2)
        pl.plot(X[ind][::10, idx[k][0]],
                X[ind][::10, idx[k][1]], '.k',ms=ms)
        pl.xlim(xlim[k])
        pl.ylim(ylim[k])
        pl.xlabel(xlab[k], fontsize=lsize)
        pl.ylabel(ylab[k], fontsize=lsize)
        if ('psf' in ylab[k]) & ('model' in ylab[k]):
            ytick = ['%0.1f' % v for v in np.linspace(-.1, 0.4, 6)]
            ytick[0] = ''
            ax.set_yticklabels(ytick)
            if i == 1:
                s = 'All'
            elif i == 3:
                s = '"Stars"'
            else:
                s = '"Galaxies"'
            ax.text(-.3, 0.5, s, ha='center', va='center', fontsize=25,
                      rotation='vertical', transform=ax.transAxes)
    f.savefig(figname, bbox_inches='tight')

def posteriors_plot(model, features, filters, figname, fgal=0.5, N=60000,
                    idx=-1, seed=123):
    """
    Plot the posterior for a star and a galaxy at r=21.
    """
    # fetch DR10 data
    X, Xcov = fetch_prepped_dr10data(N, fgal, features, filters)

    # only data within window
    dlt = 0.05
    ind = (X[:, 0] > 21 - dlt) & (X[:, 0] < 21 + dlt)
    X = X[ind]
    Xcov = Xcov[ind]

    # find stars and galaxies
    sind = np.abs(X[:, idx]) < 0.145
    gind = np.abs(X[:, idx]) > 0.145

    # pick one of each (ind = 510, 365 on my machine)
    np.random.seed(seed)
    i = np.random.randint(X[sind].shape[0])
    star = X[sind][i]
    starcov = Xcov[sind][i]
    i = np.random.randint(X[sind].shape[0])
    gal = X[gind][i]
    galcov = Xcov[gind][i]

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    # Calculate the posteriors, draw samples
    Ns = 10000
    Nd = X.shape[1]
    a, m, v = model.posterior(star.reshape(1, Nd), starcov.reshape(1, Nd, Nd))
    star_post = model.sample(a[0], m[0], v[0], size=Ns)
    a, m, v = model.posterior(gal.reshape(1, Nd), galcov.reshape(1, Nd, Nd))
    gal_post = model.sample(a[0], m[0], v[0], size=Ns)

    # fig parms
    fs = 5
    ms1, mew1 = 8, 2
    ms2, mew2 = 12, 3
    nb = 5
    lw = 2
    lsize = 20
    fac = 1.2
    idx = [[0, -1], [2, 3], [3, 4]]
    bins = [50, 50, 50]
    xlab = ['psfmag $r$', 'modelmag $g-r$', 'modelmag $r-i$']
    ylab = ['psfmag - modelmag $r$', 'modelmag $r-i$', 'modelmag $i-z$']

    # figure
    f = pl.figure(figsize=(3 * fs, 2 * fs))
    X = [star_post, gal_post]
    pl.subplots_adjust(wspace=0.3)
    for i in range(len(X)):
        if i == 0:
            X = star
            Xcov = starcov
            P = star_post
        else:
            X = gal
            Xcov = galcov
            P = gal_post
        for j in range(len(idx)):
            ax = pl.subplot(2, 3, 3 * i + j + 1)
            post = np.vstack((P[:,idx[j][0]], P[:,idx[j][1]]))
            mu = np.mean(post, axis=1)
            cov = np.cov(post)
            pl.plot(mu[0], mu[1], 'ks', ms=ms1, mew=mew1)
            pl.plot(X[idx[j][0]], X[idx[j][1]], 'k+', ms=ms2, mew=mew2)
            error_ellipse(mu, cov, ax=ax, lw=2)
            error_ellipse(X[idx[j]], Xcov[idx[j]][:,idx[j]], ax=ax, lw=2, 
                          ls='dashed')
            d = np.sqrt(np.diag(Xcov))
            mn = X[idx[j][0]] - fac * d[idx[j][0]]
            mx = X[idx[j][0]] + fac * d[idx[j][0]]
            pl.xlim((mn, mx))
            mn = X[idx[j][1]] - fac * d[idx[j][1]]
            mx = X[idx[j][1]] + fac * d[idx[j][1]]
            pl.ylim((mn, mx))
            pl.xlabel(xlab[j], fontsize=lsize)
            pl.ylabel(ylab[j], fontsize=lsize)
            pl.locator_params(nbins=nb)
    f.savefig(figname, bbox_inches='tight')

def xx_plot(epoch, model, features, filters, figname, fgal=0.5):
    """
    Plot the single epoch and xd posts versus coadd
    """
    # fetch Stripe 82 data
    X, Xcov = fetch_prepped_s82data(epoch, fgal, features, filters)
    Xcoadd, Xcoaddcov = fetch_prepped_s82data(epoch, fgal, features,
                                              filters, use_single=False)
    N = 20000
    X = X[:N]
    Xcov = Xcov[:N]
    Xcoadd = Xcoadd[:N]
    Xcoaddcov = Xcoaddcov[:N]

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    # Calculate the posteriors, draw samples
    a, m, v = model.posterior(X, Xcov)
    posts = np.zeros_like(X)
    for i in range(X.shape[0]):
        posts[i] = model.sample(a[i], m[i], v[i], size=1)

    lo = [0.01, 0.02, 0.06]
    hi = [0.99, 0.96, 0.98]
    idx = [0, 1, 4]
    bins = [100, 100, 300]
    label = ['psfmag $r$', 'modelmag $u-g$', 'modelmag $i-z$']
    N = len(idx)
    fs = 5
    lsize = 20
    f = pl.figure(figsize=(N * fs, 2 * fs))
    pl.subplots_adjust(wspace=0.3)
    for i in range(N):
        x = X[:, idx[i]]
        y = Xcoadd[:, idx[i]]
        p = posts[:, idx[i]]
        ind = (y > -999) & (Xcoaddcov[:, idx[i]][:, idx[i]] < 10.)
        x = x[ind]
        y = y[ind]
        p = p[ind]
        ax = pl.subplot(2, N, i + 1)
        v = np.sort(x)
        mn, mx = v[np.int(lo[i] * x.shape[0])], v[np.int(hi[i] * x.shape[0])]
        hist2d(x, y, ax=ax, bins=bins[i], plot_contours=True,
               plot_datapoints=True)
        pl.plot([mn, mx], [mn, mx], 'r', lw=2)
        pl.ylabel('Coadd ' + label[i], fontsize=lsize)
        pl.xlabel('Single Epoch ' + label[i], fontsize=lsize)
        pl.xlim(mn, mx)
        pl.ylim(mn, mx)
        ax = pl.subplot(2, N, i + 4)
        hist2d(p, y, ax=ax, bins=bins[i], plot_contours=True,
               plot_datapoints=True)
        pl.plot([mn, mx], [mn, mx], 'r', lw=2)
        pl.xlim(mn, mx)
        pl.ylim(mn, mx)
        pl.ylabel('Coadd ' + label[i], fontsize=lsize)
        pl.xlabel('XD Posterior ' + label[i], fontsize=lsize)
    f.savefig(figname, bbox_inches='tight')

def glob_cmd(model, globular_name, features, filters, figname, mag=0,
             color={2:(-4, -3)}, xlim=(-0.5, 2), ylim=(23, 16)):
    """
    Plot the CMD for an SDSS cluster, both raw data and XD posteriors.
    """
    # fetch data for the cluster
    X, Xcov = fetch_glob_data(globular_name, features, filters)

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    # Calculate the posteriors, record medians
    a, m, v = model.posterior(X, Xcov)
    posts = np.zeros_like(X)
    for i in range(X.shape[0]):
        posts[i] = np.median(model.sample(a[i], m[i], v[i], size=1000),
                             axis=0)

    # Get PSF mags and colors
    k = color.keys()[0]
    orig_r = X[:, mag]
    orig_gr = X[:, k] + X[:, color[k][0]] + X[:, color[k][1]]
    post_r = posts[:, mag]
    post_gr = posts[:, k] + posts[:, color[k][0]] + posts[:, color[k][1]]
    
    fs = 5
    ms = 2
    size = 15
    lsize = 20
    f = pl.figure(figsize=(2 * fs, fs))
    for i in range(1, 3):
        if i == 1:
            gr = orig_gr
            r = orig_r
            l = 'SDSS DR10'
        else:
            gr = post_gr
            r = post_r
            l = 'XD Posterior'
        ax = pl.subplot(1, 2, i)
        pl.plot(gr, r, 'ko', ms=ms, alpha=0.5)
        pl.xlim(xlim)
        pl.ylim(ylim)
        pl.xlabel('psfmag $g - r$', fontsize=lsize)
        pl.ylabel('psfmag $r$', fontsize=lsize)
        ax.text(0.95, 0.95, l, va='top', ha='right', fontsize=size,
                transform=ax.transAxes)
    f.savefig(figname, bbox_inches='tight')

def misclass_plot(epoch, model, features, filters, figname, fgal=0.5, idx=-1):
    """
    Plot the single epoch and xd posts versus coadd
    """
    # fetch Stripe 82 data
    X, Xcov = fetch_prepped_s82data(epoch, fgal, features, filters)
    Xcoadd, Xcoaddcov = fetch_prepped_s82data(epoch, fgal, features,
                                              filters, use_single=False)
    N = 20000
    X = X[:N]
    Xcov = Xcov[:N]
    Xcoadd = Xcoadd[:N]
    Xcoaddcov = Xcoaddcov[:N]
    ind = (Xcoaddcov[:, idx][:, idx] < 1.)  & (Xcov[:, idx][:, idx] < 1.)
    X = X[ind]
    Xcov = Xcov[ind]
    Xcoadd = Xcoadd[ind]
    Xcoaddcov = Xcoaddcov[ind]

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    # Calculate the posteriors, draw samples
    a, m, v = model.posterior(X, Xcov)
    posts = np.zeros_like(X)
    for i in range(X.shape[0]):
        posts[i] = np.median(model.sample(a[i], m[i], v[i], size=1000), axis=0)

    stol = 0.145
    ptol = 0.03
    Nbins = 12
    magbins = np.linspace(18., 22., Nbins)
    dlt = magbins[1] - magbins[0]
    s = np.zeros(Nbins)
    p = np.zeros(Nbins)
    for i in range(Nbins):
        ind = (Xcoadd[:, 0] > magbins[i] - dlt) & \
            (Xcoadd[:, 0] <= magbins[i] + dlt)
        sind = ind & (np.abs(Xcoadd[:, idx]) < 0.03)
        gind = ind & (np.abs(Xcoadd[:, idx]) > 0.03)
        ssind = sind & (np.abs(X[:, idx] > stol))
        sgind = gind & (np.abs(X[:, idx] < stol))
        psind = sind & (np.abs(posts[:, idx] > ptol))
        pgind = gind & (np.abs(posts[:, idx] < ptol))
        s[i] = 1. * len(X[ssind, 0]) + len(X[sgind, 0])
        p[i] = 1. * len(X[psind, 0]) + len(X[pgind, 0])
        s[i] /= len(X[ind, 0])
        p[i] /= len(X[ind, 0])

    fs = 5
    lsize = 20
    f = pl.figure(figsize=(fs, fs))
    pl.plot(magbins, s, 'k--', drawstyle='steps-mid', label='Single Epoch',
            lw=2)
    pl.plot(magbins, p, 'k', drawstyle='steps-mid', label='XD Posterior', lw=2)
    pl.xlabel('psfmag $r$', fontsize=lsize)
    pl.ylabel('Misclassification Rate', fontsize=lsize)
    f.savefig(figname, bbox_inches='tight')


def psfminusmodel_plot(epoch, model, features, filters, figname, fgal=0.5,
                       idx=-1):
    """
    Plot the single epoch and xd post histograms for psf-model of stars.
    """
    # fetch Stripe 82 data
    X, Xcov = fetch_prepped_s82data(epoch, fgal, features, filters)
    Xcoadd, Xcoaddcov = fetch_prepped_s82data(epoch, fgal, features,
                                              filters, use_single=False)

    # unpickle the XD model
    if type(model) == str: 
        f = open(model, 'rb')
        model = cPickle.load(f)
        f.close()

    fs = 5
    Nbins = [50, 50, 50]
    lsize = 20
    mags = [19.5, 20.5, 21.5]
    dlt = [0.15, 0.15, 0.15]
    f = pl.figure(figsize=(3 * fs, fs))
    pl.subplots_adjust(wspace=0.3)
    for i in range(len(mags)):
        ind = (Xcoadd[:, idx] < 0.03) & (Xcoadd[:, 0] > mags[i] - dlt[i])
        ind = ind & (Xcoadd[:, 0] <= mags[i] + dlt[i])

        x = X[ind]
        xc = Xcov[ind]

        a, m, v = model.posterior(x, xc)
        posts = np.zeros_like(x)
        for j in range(x.shape[0]):
            posts[j] = np.median(model.sample(a[j], m[j], v[j], size=1000),
                                 axis=0)

        ax = pl.subplot(1, 3, i + 1)
        h, b = np.histogram(x[:, idx], Nbins[i])
        d = (b[1] - b[0]) / 2.
        b = np.append([b[0] - d], b[:-1] + d)
        h = np.append([1.0], h)
        pl.plot(b, h, drawstyle='steps-mid', linestyle='dotted',
                color='k', lw=2)
        h, b = np.histogram(posts[:, idx], Nbins[i])
        d = (b[1] - b[0]) / 2.
        b = np.append([b[0] - d], b[:-1] + d)
        h = np.append([1.0], h)
        pl.plot(b, h, drawstyle='steps-mid', color='k', lw=2)
        pl.xlabel('psfmag - modelmag $r$', fontsize=lsize)
        pl.ylabel('counts', fontsize=lsize)
        ax.text(0.95, 0.95, '$r=%0.1f$' % mags[i], fontsize=lsize, ha='right',
                va='top', transform=ax.transAxes)
        pl.xlim(-0.1, 0.2)
    f.savefig(figname, bbox_inches='tight')

def s82_star_galaxy_classification(model_parms_file, epoch, Nstar,
                                   features, filters, r_pmm, figname,
                                   threshold=0., Nthreads=4):
    """
    Compare quality of classifcation for a model with the s82 coadd.  Should 
    be a model trained on s82 single epoch data.
    """
    # get the data
    single, singlecov = fetch_matched_s82data(epoch, features=features,
                                              filters=filters)
    coadd, coaddecov = fetch_matched_s82data(epoch, features=features,
                                             filters=filters, use_single=False)

    # classfy the single epoch data
    single_class = np.zeros(single.shape[0])
    ind = np.abs(single[:, r_pmm]) < 0.145 
    single_class[ind] = 1.

    alpha, mu, V, _, _ = load_xd_parms(model_parms_file)
    logls = log_multivariate_gaussian_Nthreads(single, mu, V, singlecov,
                                               Nthreads)
    logls += np.log(alpha)
    logodds = logsumexp(logls[:, :Nstar], axis=1)
    logodds -= logsumexp(logls[:, Nstar:], axis=1)
    ind = logodds > threshold
    model_class = np.zeros(single.shape[0])
    model_class[ind] = 1.

    coadd_class = np.zeros(single.shape[0])
    ind = np.abs(coadd[:, r_pmm]) < 0.03
    coadd_class[ind] = 1.

    fs = 10
    f = pl.figure(figsize=(2 * fs, 2 * fs))
    pl.subplot(221)
    pl.plot(single[single_class==0, 0], single[single_class==0, r_pmm], '.',
            color='#ff6633', alpha=0.2)
    pl.plot(single[single_class==1, 0], single[single_class==1, r_pmm], '.',
            color='#3b5998', alpha=0.2)
    pl.ylim(-0.2, 0.5)
    pl.subplot(222)
    pl.plot(coadd[single_class==0, 0], coadd[single_class==0, r_pmm], '.',
            color='#ff6633', alpha=0.2)
    pl.plot(coadd[single_class==1, 0], coadd[single_class==1, r_pmm], '.',
            color='#3b5998', alpha=0.2)
    pl.plot([17.5, 22.], [0.03, 0.03], 'k')
    pl.ylim(-0.2, 0.5)
    pl.subplot(223)
    pl.plot(single[model_class==0, 0], single[model_class==0, r_pmm], '.',
            color='#ff6633', alpha=0.2)
    pl.plot(single[model_class==1, 0], single[model_class==1, r_pmm], '.',
            color='#3b5998', alpha=0.2)
    pl.ylim(-0.2, 0.5)
    pl.subplot(224)
    pl.plot(coadd[model_class==0, 0], coadd[model_class==0, r_pmm], '.',
            color='#ff6633', alpha=0.2)
    pl.plot(coadd[model_class==1, 0], coadd[model_class==1, r_pmm], '.',
            color='#3b5998', alpha=0.2)
    pl.plot([17.5, 22.], [0.03, 0.03], 'k')
    pl.ylim(-0.2, 0.5)
    f.savefig(figname, bbox_inches='tight')

if __name__ == '__main__':
    ddir = os.environ['xddata']
    pdir = os.environ['xdplots']

    epoch = 3
    N = 240000
    K = 32
    n_iter = 256
    Nstar = 16
    data = 'dr10'
    features = ['psf_mag', 'model_colors', 'psf_minus_model']
    filters = ['r', 'ug gr ri iz', 'ugriz']
    message = 'pm_mc_pmm_r_all_all'
    message = 'pm_mc_pmm_r_all_all_v1'
    model_parm_file = ddir + '/s82_%d_%d_%d_%s.hdf5' % (N, K, Nstar, message)
    r_pmm = -3

    if True:
        s82_star_galaxy_classification(model_parm_file, epoch, Nstar,
                                       features, filters, r_pmm,
                                       pdir + 'foo.png')

    # test contour plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        contours_and_data(epoch, model, features, filters, figname, fgal=0.5,
                          idx=-3, data=data)

    # test histo plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        psfminusmodel_plot(epoch, model, features, filters, figname, idx=-3)

    # call for these might be affected by the new way model parms are saved.
    """
    # test glob cmd plot
    if False:
        gn = 'm3'
        glob_cmd(model, gn, features, filters,
                 os.environ['xdplots'] + gn + '.png')

    # test xx plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        xx_plot(epoch, model, features, filters, figname)

    # test posteriors plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        posteriors_plot(model, features, filters, figname, idx=-3)

    # test error rate plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        error_rates(epoch, model, features, filters, figname, idx=-3, N=1000)

    # test error rate plot
    if False:
        figname = os.environ['xdplots'] + 'foo.png'
        error_rates(epoch, model, features, filters, figname, idx=-3, N=1000)


    # run paper figure generation
    if True:
        fig_6()
    """
