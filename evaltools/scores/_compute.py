"""Functions designed to compute scores between two arrays."""
from datetime import time
import numpy as np
import pandas as pd
from ._scoreobj import Score
from . import IMPLEMENTED_SCORES


def _user_score_tpl(score):
    """
    Return the list of the parameters of a function.

    The function returns a string that contains
    the list of parameters of a score, formatted like so:
    "score_name {param1:type1} {param2:type2} {param3:type3}"

    It is meant to be used if a syntax error is raised when using
    a parametered score.

    Parameters
    ----------
    score : Score

    Returns
    -------
    tpl : str

    """
    param_names = [
        f"{p}: {t.__name__}" for p, t in
        score.score_func.__annotations__.items()
    ]
    params_names = '{' + '} {'.join(param_names) + '}'

    tpl = f"{score.name} {params_names}"
    return tpl


def get_score(user_score):
    """
    Get a score from its name and eventually its parameters.

    Parameters
    ----------
    user_score : str
        User string to access a given score.

    Returns
    -------
    score : evaltools._scoreobj.Score

    """
    tokens = user_score.lower().split()
    sc_name = tokens[0]

    try:
        sc_obj = IMPLEMENTED_SCORES[sc_name]
    except KeyError as e:
        raise KeyError(f"Unknown score {e}")

    if not sc_obj._have_extra_params:
        return sc_obj

    params_types = sc_obj.score_func.__annotations__
    defaults = sc_obj.score_func.__defaults__ or []
    sc_params = {}.fromkeys(params_types)

    if len(tokens[1:]) == 0 and len(params_types) == len(defaults):
        tokens.extend([val for val in defaults])

    if not (len(params_types) == len(tokens[1:])):
        raise ValueError(
            "Invalid score string provided : " +
            f"Wrong number of parameters for {sc_name}\n"
            f"expected : {_user_score_tpl(sc_obj)}"
        )

    try:
        for (p, ptype), token in zip(params_types.items(), tokens[1:]):
            sc_params[p] = ptype(token)
    except ValueError as e:
        raise ValueError(f"Invalid score string provided for {sc_name} : {e}")

    return Score(
        func=lambda obs, sim: sc_obj.score_func(obs, sim, **sc_params),
        name=sc_name,
        units=sc_obj.units
    )


def stats2d_numpy(obs, sim, score_list, threshold, **kwargs):
    """
    Compute scores from 2D numpy arrays along a given axis.

    Parameters
    ----------
    obs : numpy.ndarray
        Array of observations.
    sim : numpy.ndarray
        Array of simulations.
    score_list : list of str
        List of scores to compute.
    threshold : int
        Minimal number of data available in both obs
        and sim required per column.

    Returns
    -------
    numpy.ndarray
        Array with one column per score and rows corresponding to columns
        of the input arrays.

    """
    score_funcs = {}
    for sc in score_list:
        score_funcs[sc] = get_score(sc)

    res = np.full((obs.shape[1], len(score_list)), np.nan)
    for col in range(obs.shape[1]):
        idx = np.logical_not(np.isnan(obs[:, col]) | np.isnan(sim[:, col]))
        if np.sum(idx) >= threshold:
            obs_without_nan = obs[:, col][idx]
            sim_without_nan = sim[:, col][idx]
            for i, sc in enumerate(score_list):
                res[col, i] = score_funcs[sc](
                    obs_without_nan,
                    sim_without_nan,
                    **kwargs,
                )
    return res


def stats2d(
        obs, sim, score_list, axis=0, threshold=0.75,
        **kwargs):
    """
    Compute scores from 2D DataFrames along a given axis.

    Parameters
    ----------
    obs : pandas.DataFrame
        DataFrame of observations.
    sim : pandas.DataFrame
        DataFrame of simulations.
    score_list : list of str
        List of scores to compute.
    axis : 0 or 1
        Axis along which to compute the scores.
    threshold : int or float
        Minimal number (if type(threshold) is int) or minimal rate
        (if type(threshold) is float) of data available in both obs and
        sim required per column (if axis == 0) or per row (if axis == 1) to
        compute the scores.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one column per score and rows equal to columns
        (if axis == 0) or rows (if axis == 1) of input DataFrames.

    """
    if axis == 1:
        obs = obs.T
        sim = sim.T

    # define threshold value if it is given as a ratio
    if type(threshold) is float:
        if threshold >= 1.:
            print("*** Warning: threshold of type float >= 1")
        threshold = threshold*obs.shape[0]

    score_array = stats2d_numpy(obs.values, sim.values, score_list=score_list,
                                threshold=threshold, **kwargs)

    return pd.DataFrame(score_array, index=obs.columns, columns=score_list)


def ft_stats(
        obs, sim, score_list, availability_ratio=0.75, day=0, step=1,
        **kwargs):
    """
    Compute forecast time scores for each station.

    Parameters
    ----------
    obs : pandas.DataFrame
        DataFrame with datetime index and one column per station.
    sim : pandas.DataFrame
        DataFrame with datetime index and one column per station.
    score_list : list of str
        List of scores to compute.
    availability_ratio : float
        Minimal rate of data (computed scores per time) available on the
        period required per forecast time to compute the mean scores.
    day : int
        The forecast day corresponding to the simulated data.
    step : int
        Time step in hours.

    Returns
    -------
    dictionary
        Dictionary with keys corresponding to computed scores, that contains
        pandas.DataFrames with one row per forecast time and one column per
        station. For example, rows will be 24, 25, ..., 47 if day = 1.

    """
    # copy obs and sim variables to avoid modifying the original objects
    obs = obs.copy(deep=True)
    sim = sim.copy(deep=True)

    obs.index = pd.MultiIndex.from_arrays([obs.index.date, obs.index.time],
                                          names=['Date', 'Time'])
    sim.index = pd.MultiIndex.from_arrays([sim.index.date, sim.index.time],
                                          names=['Date', 'Time'])
    res = {}
    for s in score_list:
        res[s] = pd.DataFrame()
    for t in range(0, 24, step):
        o = obs.xs(time(t), level='Time', axis=0)
        s = sim.xs(time(t), level='Time', axis=0)
        st = stats2d(
            o, s, score_list,
            axis=0,
            threshold=float(availability_ratio),
            **kwargs,
        )
        for s in score_list:
            res[s][t+24*day] = st[s]
    return res


def contingency_table(obs, sim, thr):
    """
    Contigency table.

    Built from a vector of observations and a vector simulations
    compared to a threshold. The contingency table counts the
    number of values greater than the threshold. For example,
    tab[1, 0] is equal to the number of stations where observations
    are lower than the threshold but simulated values are greater
    than the threshold.

    Parameters
    ----------
    obs : numpy array
        Array containing observations values.
    sim : numpy array
        Array containing simulation values.
    thr : float
        Threshold.

    Returns
    -------
    tab : 3x3 numpy array
        tab[2, 0:1] and tab[0:1, 2] contain
        respectively the sum along the lines and the sum along
        the columns of the contingency table.

    """
    # drop nan
    idx = np.logical_not(np.logical_or(np.isnan(obs), np.isnan(sim)))
    obs_without_nan = obs[idx]
    sim_without_nan = sim[idx]

    # contingency table
    n = len(obs_without_nan)
    tab = np.zeros((3, 3))
    idx_obs = obs_without_nan > thr
    idx_sim = sim_without_nan > thr
    tab[0, 0] = np.sum(np.logical_and(idx_obs, idx_sim))
    tab[1, 0] = np.sum(np.logical_and(np.logical_not(idx_obs), idx_sim))
    tab[0, 1] = np.sum(np.logical_and(idx_obs, np.logical_not(idx_sim)))
    tab[1, 1] = n - np.sum(tab)
    tab[2, 0] = np.sum(tab[:, 0])
    tab[2, 1] = np.sum(tab[:, 1])
    tab[0, 2] = np.sum(tab[0, :])
    tab[1, 2] = np.sum(tab[1, :])
    tab[2, 2] = np.sum(tab[2, :])
    return tab


def exceedances_scores(obs, sim, thr):
    """Compute several scores based on the contingency table."""
    idx_nan = np.logical_or(np.isnan(obs), np.isnan(sim))
    o = obs[~idx_nan]
    s = sim[~idx_nan]
    tab = contingency_table(o, s, thr)
    n = tab[2, 2]
    # Calcul des scores
    resul = {}
    if n != 0:
        # Taux de bonnes previsions
        resul['accuracy'] = (tab[0, 0] + tab[1, 1]) / float(n)
        # Bias score
        if tab[2, 0] * tab[0, 2] != 0:
            resul['bias_score'] = tab[2, 0] / float(tab[0, 2])
        # Success ratio
        if tab[0, 0] * tab[2, 0] != 0:
            resul['success_ratio'] = tab[0, 0] / float(tab[2, 0])
        # probability of detection (Hit rate)
        if tab[0, 0] * tab[0, 2] != 0:
            resul['hit_rate'] = tab[0, 0] / float(tab[0, 2])
        # false alarm ratio
        if tab[1, 0] * tab[2, 0] != 0:
            resul['false_alarm_ratio'] = tab[1, 0] / float(tab[2, 0])
        # probability of false detection = false alarm rate
        if tab[1, 0] * tab[1, 2] != 0:
            resul['prob_false_detect'] = tab[1, 0] / float(tab[1, 2])
        # Threat Score
        if tab[0, 2] + tab[1, 0] != 0 and tab[0, 0] != 0:
            resul['threat_score'] = tab[0, 0] / float(tab[0, 2] + tab[1, 0])
        # Equitable Threat Score
        alea = tab[0, 2] * tab[2, 0] / float(n)
        if tab[0, 2] + tab[1, 0] - alea != 0:
            resul['equitable_ts'] = (tab[0, 0] - alea)/float(tab[0, 2] +
                                                             tab[1, 0] - alea)
        # Peirce Skill Score (Hanssen and Kuipers discriminant)
        if tab[0, 2] * tab[1, 2] != 0:
            resul['peirce_ss'] = ((tab[0, 0] / float(tab[0, 2])) -
                                  (tab[1, 0] / float(tab[1, 2])))
        # Heidke Skill Score
        alea = (tab[0, 2] * tab[2, 0] + tab[1, 2] * tab[2, 1]) / float(n)
        if n - alea != 0:
            resul['heidke_ss'] = ((tab[0, 0] + tab[1, 1] - alea) /
                                  float(n - alea))
        # Rousseau Skill Score
        alea = ((tab[0, 2] / 2. + tab[2, 0] / 2.) ** 2 +
                (tab[1, 2] / 2. + tab[2, 1] / 2.) ** 2) / float(n)
        if n - alea != 0:
            resul['rousseau_ss'] = ((tab[0, 0] + tab[1, 1] - alea) /
                                    float(n - alea))
        # Odds Ratio Skill Score
        if tab[0, 0] * tab[1, 1] * tab[1, 0] * tab[0, 1] != 0:
            th = tab[0, 0] * tab[1, 1] / float(tab[1, 0] * tab[0, 1])
            resul['odds_ratio'] = th
            if th + 1 != 0:
                resul['odds_ratio_ss'] = (th - 1) / (th + 1)
    return resul, tab
