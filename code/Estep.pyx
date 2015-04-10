import cython
import numpy as np
cimport numpy as np

from cython_gsl cimport *

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef gsl_matrix_set_from_Num(gsl_matrix * m1, Nm1):
    cdef int i,j
    for i from 0 <= i < Nm1.shape[0]:
        for j from 0 <= j < Nm1.shape[1]:
            gsl_matrix_set(m1,i, j, Nm1[i,j])

cdef Num_set_from_gsl_matrix(gsl_matrix * m1, Nm1):
    cdef int i,j
    for i from 0 <= i < Nm1.shape[0]:
        for j from 0 <= j < Nm1.shape[1]:
            Nm1[i,j] = gsl_matrix_get(m1,i, j)

cdef invlogdet(gsl_matrix *m, gsl_matrix *inverse, gsl_permutation *p):
    cdef int s
    cdef double lndet
    gsl_linalg_LU_decomp(m, p, &s)
    lndet = gsl_linalg_LU_lndet(m)
    gsl_linalg_LU_invert(m, p, inverse)
    return lndet

def _Estep(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] mu,
           np.ndarray[DTYPE_t, ndim=3] V, np.ndarray[DTYPE_t, ndim=3] C,
           np.ndarray[DTYPE_t, ndim=1] alpha):
    """
    E-step for XD.  This is just a hair faster than a numpy implementation
    but uses a about half the memory.
    """
    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef int K = mu.shape[0]

    cdef double chi2, norm
    cdef double const = -0.5 * D * np.log(2 * np.pi)

    p = gsl_permutation_alloc(D)
    cdef gsl_matrix *gslcov = gsl_matrix_alloc(D, D)
    cdef gsl_matrix *gslicov = gsl_matrix_alloc(D, D)

    cdef np.ndarray[DTYPE_t, ndim=2] cov  = np.zeros((D, D))
    cdef np.ndarray[DTYPE_t, ndim=2] icov = np.zeros((D, D))
    cdef np.ndarray[DTYPE_t, ndim=2] qs = np.zeros((N, K))
    cdef np.ndarray[DTYPE_t, ndim=3] bs = np.zeros((N, K, D))
    cdef np.ndarray[DTYPE_t, ndim=4] Bs = np.zeros((N, K, D, D))

    for i in range(N):
        norm = 0.0
        for j in range(K):
            # invert T
            cov = V[j] + C[i]
            gsl_matrix_set_from_Num(gslcov, cov)
            logdet = invlogdet(gslcov, gslicov, p)
            Num_set_from_gsl_matrix(gslicov, icov)

            # calculate qs
            xmm = x[i] - mu[j]
            chi2 = np.dot(xmm, np.dot(icov, xmm.T))
            qs[i, j] = np.exp(const - 0.5 * (logdet + chi2)) * alpha[j]
            norm += qs[i, j]

            # calculate little b
            bs[i, j] = np.dot(V[j], np.dot(icov, xmm.T)) + mu[j]

            # calculate big B
            Bs[i, j] = V[j] - np.dot(V[j], np.dot(icov, V[j]))

        qs[i] /= norm

    gsl_matrix_free(gslcov)
    gsl_matrix_free(gslicov)
    gsl_permutation_free(p)
    return qs, bs, Bs
