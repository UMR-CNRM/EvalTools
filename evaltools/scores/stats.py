# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
Scores and statistics used by evaltools.

This Module containing the implementation of all the scores and statistics
used by evaltools.

To add a new score, use the `score` decorator
and please comply to the following rules :

    1. Always give a name and a unit to your score
    2. The name of the first two arguments need to be 'obs' and 'sim' and need
       to be numpy arrays
    3. if there's any other parameter, it needs to be annotated with a python
       type

"""
import numpy as np
import scipy.stats

from . import _fastimpl as fst
from ._scoreobj import score
from . import _sort_keys as sk


def percentile(x, perc):
    """
    Shorthand for numpy nanpercentile.

    Parameters
    ----------
    x : numpy.ndarray
    perc : float (between 0 and 1)

    Returns
    -------
        The *perc*th percentile.

    """
    return np.nanpercentile(x, perc*100, axis=0)


@score(name="NbObs", units="{field_units}")
def nb_obs(obs, sim):
    """Return the number of observations."""
    return len(obs)


@score(name="MeanBias", units="{field_units}", alias="Bias",
       sort_key=sk.absolute)
def mean_bias(obs, sim):
    """Mean bias."""
    return fst.mean_bias(obs, sim)


@score(name="mean_bias_Ttest", units="{field_units}", sort_key=sk.absolute)
def mean_bias_t_test(obs, sim):
    """
    Mean Bias computing between two numpy arrays.

    Return a bias set to 0 when the T-test for the null
    hypothesis that observations and simulations have identical average
    values is not significant (pvalue > 0.05).

    """
    b = mean_bias(obs, sim)
    pvalue = scipy.stats.mannwhitneyu(obs, sim)[1]
    if pvalue > 0.05:
        bias_t = 0.
    else:
        bias_t = b
    return bias_t


@score(name="MMB", units=None, alias='MNMB', sort_key=sk.absolute)
def modified_mean_bias(obs, sim):
    """Return the Modified Mean bias."""
    return fst.modified_mean_bias(obs, sim)


@score(name="FracBias", units=None, sort_key=sk.absolute)
def fractional_bias(obs, sim):
    """Fractional bias."""
    return fst.fractional_bias(obs, sim)


@score(name="BiasPct", units=None, sort_key=sk.absolute,
       alias="relative_mean_bias")
def relative_mean_bias(obs, sim):
    """Relative mean bias."""
    return fst.relative_mean_bias(obs, sim)


@score(name="Ratio", units=None, sort_key=sk.abs_to_one,
       alias="variances_ratio")
def variances_ratio(obs, sim):
    """Return the Variance ratio."""
    return fst.variances_ratio(obs, sim)


@score(name="Ratio+t", units=None, sort_key=sk.abs_to_one)
def variances_ratio_l_test(obs, sim):
    """
    Compute simulation variance / observation variance ratio.

    Return a variance ratio set to 0 when the Levene test for the null
    hypothesis that observations and simulations are from populations
    with equal variances is not significant (pvalue > 0.05).

    """
    return fst.variances_ratio(obs, sim, test=True)


@score(name="SDratio", units=None, sort_key=sk.abs_to_one, alias="std_ratio")
def sd_ratio(obs, sim):
    """Return the standard deviation ratio."""
    return np.sqrt(fst.variances_ratio(obs, sim))


@score(name="MAE", units="{field_units}", sort_key=sk.identity)
def mae(obs, sim):
    """Return the Mean Absolute Error."""
    return fst.MAE(obs, sim)


@score(name="RMSE", units="{field_units}", sort_key=sk.identity)
def rmse(obs, sim):
    """Return the Root Mean Squared Error."""
    return fst.RMSE(obs, sim)


@score(name="FGE", units=None, sort_key=sk.identity)
def fge(obs, sim):
    """Return the Fractional Gross Error."""
    return fst.fge(obs, sim)


@score(name="CRMSE", units="{field_units}", sort_key=sk.identity)
def crmse(obs, sim):
    """Return the Centered Root Mean Squared Error."""
    return fst.CRMSE(obs, sim)


@score(name="NMSE", units=None, sort_key=sk.identity)
def nmse(obs, sim):
    """Return the Normalized Mean Squared."""
    return fst.NMSE(obs, sim)


def correlation(obs, sim, mtd='Pearson'):
    """Correlation computing."""
    if np.std(obs) > 0 and np.std(sim) > 0:
        if mtd == 'Pearson':
            corr, p = scipy.stats.pearsonr(obs, sim)
        elif mtd == 'Spearman':
            corr, p = scipy.stats.spearmanr(obs, sim)
    else:
        corr, p = np.nan, np.nan
    if np.isnan(corr) or np.isnan(p):
        corr = np.nan
        p = np.nan
    corr_t = corr*(p < 0.05)
    return corr, corr_t


@score(name="PearsonR", units=None, sort_key=sk.negate, alias='correlation')
def pearsonr(obs, sim):
    """Return the Pearson's correlation."""
    return fst.PearsonR(obs, sim)


@score(name="PearsonR+t", units=None, sort_key=sk.negate)
def pearsonr_t(obs, sim):
    """Return the Pearson's correlation."""
    return correlation(obs, sim, mtd='Pearson')[1]


@score(name="SpearmanR", units=None, sort_key=sk.negate)
def spearmanr(obs, sim):
    """Return the Spearman's correlation."""
    return correlation(obs, sim, mtd='Spearman')[0]


@score(name="SpearmanR+t", units=None, sort_key=sk.negate)
def spearmanr_t(obs, sim):
    """Return the Spearman's correlation."""
    return correlation(obs, sim, mtd='Spearman')[1]


@score(name="FactOf2", units=None, sort_key=sk.negate)
def factor_of_two(obs, sim):
    """Return the factor of two."""
    return fst.factor_of_two(obs, sim)


@score(name="sim_median", units="{field_units}")
def sim_median(obs, sim):
    """Return the Median of modelled values."""
    return np.median(sim)


@score(name="obs_median", units="{field_units}")
def obs_median(obs, sim):
    """Return the Median of observed values."""
    return np.median(obs)


@score(name="sim_mean", units="{field_units}")
def sim_mean(obs, sim):
    """Return the Mean of modelled values."""
    return np.mean(sim)


@score(name="obs_mean", units="{field_units}")
def obs_mean(obs, sim):
    """Return the Mean of observed values."""
    return np.mean(obs)


@score(name="obs_std", units="{field_units}")
def obs_std(obs, sim):
    """Return the Standard deviation of the observed values."""
    return np.std(obs)


@score(name="sim_std", units="{field_units}")
def sim_std(obs, sim):
    """Return the Standard deviation of the modelled values."""
    return np.std(sim)


@score(name="bias_std", units="{field_units}", alias="std(obs-sim)")
def bias_std(obs, sim):
    """Return the Standard deviation of the bias."""
    return np.std(obs - sim)


@score(name="obs_percentile", units="{field_units}")
def obs_percentile(obs, sim, perc: float):
    """
    Return the *perc* percentile of the observed values.

    Parameters
    ----------
    perc : float

    """
    return percentile(obs, perc)


@score(name="sim_percentile", units="{field_units}")
def sim_percentile(obs, sim, perc: float):
    """
    Return the *perc* percentile of modelled values.

    Parameters
    ----------
    perc : float

    """
    return percentile(sim, perc)


@score(name="success_rate", units=None, sort_key=sk.negate)
def success_rate(obs, sim, tolerance_thr: float, utility_thr: float):
    """
    Success rate with tolerance.

    Compute the rate of values where the absolute difference between the
    observed value and the simulated one is lower than the tolerance threshold.
    If the absolute difference between the observed value and the simulated one
    lays between the tolerance threshold and the utility threshold, the success
    score of the simulated value is defined with a linear function equal to 1
    if |obs - sim| = tolerance_thr and equal to 0 if |obs - sim| = utility_thr.

    Parameters
    ----------
    obs : numpy.ndarray
        Array of observations.
    sim : numpy.ndarray
        Array of simulations.
    tolerance_thr : scalar
        Tolerance threshold.
    utility_thr : scalar
        Utility threshold.

    """
    err = np.abs(obs - sim)

    rate = np.mean(
        np.where(err <= tolerance_thr, 1, 0) +
        np.where(
            np.logical_and(err > tolerance_thr, err < utility_thr),
            (utility_thr - err) / (utility_thr - tolerance_thr),
            0,
        )
    )

    return rate


@score(
    name="kendalltau", units=None, alias='kendall_correlation',
    sort_key=sk.negate,
)
def kendal_correlation(obs, sim):
    """Return the Kendall’s tau."""
    return scipy.stats.kendalltau(obs, sim).correlation


@score(name="nrmse", units=None, sort_key=sk.identity, alias="rrmse")
def nrmse(obs, sim, norm: str = 'mean'):
    """
    Return the Normalized Root Mean Squared Error.

    Parameters
    ----------
    norm : str
        Norm operator. Possible values are

            - 'obs_mean' : mean of observations values
            - 'sim_mean' : mean of simulations values
            - 'std_obs' : standard deviation of observations
            - 'spread' : difference between maximum and minimum in observations
            - 'iqr' : interquartile range (Q3 - Q1)

    """
    if norm in ['mean', 'obs_mean']:
        div = np.mean(obs)
    elif norm == 'sim_mean':
        div = np.mean(sim)
    elif norm == 'obs_std':
        div = np.std(obs)
    elif norm == 'spread':
        div = np.max(obs) - np.min(obs)
    elif norm == 'iqr':
        div = np.percentile(obs, 75) - np.percentile(obs, 25)
    else:
        raise ValueError(
            "Argument norm must equal to 'obs_mean', 'sim_mean', 'std_obs' "
            "'spread' or 'iqr'."
        )

    return fst.RMSE(obs, sim)/div
