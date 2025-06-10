# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
cimport numpy as np
cimport cython
from libc.math cimport abs, sqrt

import numpy as np
from scipy.stats import levene


cdef double mean1D(double[:] vect):
    """1D array mean computing."""
    cdef Py_ssize_t vect_len = vect.shape[0]
    cdef double vect_mean = 0
    cdef Py_ssize_t x
    for x in range(vect_len):
        vect_mean += vect[x]
    vect_mean /= vect_len
    return vect_mean


cdef double variance1D(double[:] vect):
    """1D array mean computing."""
    cdef Py_ssize_t vect_len = vect.shape[0]
    cdef double var_vect = 0
    cdef Py_ssize_t x
    cdef double m_vect = mean1D(vect)

    for x in range(vect_len):
        var_vect += (vect[x] - m_vect)**2
    var_vect /= vect_len

    return var_vect


cpdef double mean_bias(double[:] obs, double[:] sim):
    """Mean Bias computing between two numpy arrays."""
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef double obs_mean = 0
    cdef double sim_mean = 0
    cdef Py_ssize_t x

    for x in range(x_max):
        obs_mean += obs[x]
        sim_mean += sim[x]

    obs_mean /= x_max
    sim_mean /= x_max

    return sim_mean - obs_mean


cpdef double relative_mean_bias(double[:] obs, double[:] sim):
    """Relative Mean Bias computing between two numpy arrays."""
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef double obs_mean = 0
    cdef double sim_mean = 0
    cdef Py_ssize_t x

    for x in range(x_max):
        obs_mean += obs[x]
        sim_mean += sim[x]

    obs_mean /= x_max
    sim_mean /= x_max

    if obs_mean != 0.:
        return (100.*(sim_mean - obs_mean))/obs_mean
    else:
        return 100.


cpdef double fractional_bias(double[:] obs, double[:] sim):
    """Fractional bias computing."""
    cdef double m_obs = mean1D(obs)
    cdef double m_sim = mean1D(sim)
    if (m_sim + m_obs) != 0:
        return 100 * 2 * (m_sim - m_obs) / (m_sim + m_obs)
    else:
        return 0


cpdef double modified_mean_bias(double[:] obs, double[:] sim):
    """Modified Mean Bias computing."""
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef Py_ssize_t x
    cdef double ratio_sum = 0
    cdef double mmb

    for x in range(x_max):
        if (obs[x] + sim[x]) != 0.:
            ratio_sum += (sim[x] - obs[x])/(obs[x] + sim[x])

    mmb = 2*ratio_sum/x_max

    return mmb


cpdef double fge(double[:] obs, double[:] sim):
    """
    Fractional Gross Error computing.

    Parameters
    ----------
    obs : numpy 1D array
        Array containing observations values.
    sim : numpy 1D array
        Array containing simulation values.

    Returns
    -------
        Fractional Gross Error.

    """
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef Py_ssize_t x
    cdef double ratio_sum = 0
    cdef double mmb

    for x in range(x_max):
        if (obs[x] + sim[x]) != 0.:
            ratio_sum += abs((sim[x] - obs[x])/(obs[x] + sim[x]))

    mmb = 2*ratio_sum/x_max

    return mmb


cpdef double variances_ratio(double[:] obs, double[:] sim, bint test=False):
    """Variance ratio and Levene test."""
    cdef double obs_var = variance1D(obs)
    cdef double sim_var
    cdef double ratio
    cdef double ratioT
    if obs_var != 0.:
        sim_var = variance1D(sim)
        ratio = sim_var/obs_var
        if test is True:
            ltest = levene(obs, sim)[1] < 0.05
            ratioT = ratio*ltest + 1*~ltest
            return ratioT
        else:
            return ratio
    else:
        return 0.


cpdef double MAE(double[:] obs, double[:] sim):
    """Mean absolute error computing."""
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef double mae = 0
    cdef Py_ssize_t x
    for x in range(x_max):
        mae += abs(sim[x] - obs[x])
    mae /= x_max
    return mae


cpdef double RMSE(double[:] obs, double[:] sim):
    """RMSE computing."""
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef double mse = 0
    cdef Py_ssize_t x
    for x in range(x_max):
        mse += (sim[x] - obs[x])**2
    mse /= x_max
    return sqrt(mse)


cpdef double CRMSE(double[:] obs, double[:] sim):
    """CRMSE computing."""
    cdef double m_obs = mean1D(obs)
    cdef double m_sim = mean1D(sim)
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef double crmse = 0
    cdef Py_ssize_t x
    for x in range(x_max):
        crmse += ((sim[x]-m_sim)-(obs[x]-m_obs))**2
    crmse /= x_max
    crmse = sqrt(crmse)
    return crmse


cpdef double NMSE(double[:] obs, double[:] sim):
    """NMSE computing."""
    cdef double m_obs = mean1D(obs)
    cdef double m_sim = mean1D(sim)
    cdef double mean_prod = m_sim*m_obs
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef Py_ssize_t x
    cdef double mse = 0
    if mean_prod != 0:
        for x in range(x_max):
            mse += (sim[x] - obs[x])**2
        mse /= x_max
        return mse/mean_prod
    else:
        return 0


cpdef double PearsonR(double[:] obs, double[:] sim):
    """Pearson Correlation computing."""
    cdef double m_obs = mean1D(obs)
    cdef double m_sim = mean1D(sim)
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef Py_ssize_t x
    cdef double std_obs = 0.
    cdef double std_sim = 0.
    cdef double cov = 0.
    for x in range(x_max):
        std_obs += (obs[x]-m_obs)**2
        std_sim += (sim[x]-m_sim)**2
        cov += (sim[x]-m_sim)*(obs[x]-m_obs)
    std_obs = sqrt(std_obs)
    std_sim = sqrt(std_sim)
    if std_obs > 0. and std_sim > 0.:
        return cov/(std_obs*std_sim)
    else:
        return np.nan


cpdef double factor_of_two(double[:] obs, double[:] sim):
    """Factor of two."""
    cdef double cptr = 0.
    cdef double total = 0.
    cdef Py_ssize_t x_max = obs.shape[0]
    cdef Py_ssize_t x
    for x in range(x_max):
        if obs[x] == 0. and sim[x] == 0.:
            cptr += 1
            total += 1
        elif obs[x] != 0:
            total += 1
            if 0.5 <= sim[x] / obs[x] <= 2:
                cptr += 1
    if total > 0:
        return cptr/total
    else:
        return 0.
