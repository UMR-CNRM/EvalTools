# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers functions designed to compute tables of scores."""

from functools import partial
import numpy as np
import pandas as pd

import evaltools as evt


def average_scores(
        objects, forecast_day, score_list,
        score_type='temporal', averaging='median',
        title=None, labels=None,
        availability_ratio=0.75, min_nb_sta=10,
        output_file=None, output_latex=None, float_format=None):
    """
    Build a table with average values of temporal or spatial scores.

    This function is based on Evaluator.temporal_scores and
    Evaluator.spatial_scores methods.
    Scores are first computed for each station (score_type = 'temporal')
    or for each time (score_type = 'spatial'). Then, the median or mean
    of these scores is taken.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for computation.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.
    score_list : list of str
        List of computed scores.
    score_type : str
        Computing method selected among 'temporal' or 'spatial'.
    averaging : str
        Type of score averaging selected from 'mean' or 'median'.
    title : str or None
        Title to add in the output file.
    labels : None or list of str
        Labels of the objects.
    availability_ratio : float
        Minimal rate of data available on the period required to compute the
        temporal scores (if score_type = 'temporal') or to compute the
        temporal average (if score_type = 'spatial').
    min_nb_sta : int
        Minimal number of stations required to compute the spatial average (if
        score_type = 'temporal') or to compute the spatial score (if
        score_type = 'spatial').
    output_file : str
        File where to save the table. If None, the table is printed.
    output_latex : str or None
        If not None, file where to save the table in a LaTeX layout.
    float_format : str
        String format for table values.

    """
    # defining the averaging function
    if score_type == 'temporal':
        sc_function = evt.Evaluator.average_temporal_scores
    elif score_type == 'spatial':
        sc_function = evt.Evaluator.average_spatial_scores
    else:
        raise evt.EvaltoolsError(
            "score_type argument must be either equal to "
            "'temporal' or 'spatial'."
        )

    # build 'labels' from objects 'model' attribute if 'labels' is None
    if labels is None:
        labels = [obj.model for obj in objects]

    # compute stats
    stats = pd.DataFrame(columns=score_list, dtype=float)
    for obj, lab in zip(objects, labels):
        res = sc_function(
            obj,
            score_list=score_list,
            availability_ratio=availability_ratio,
            min_nb_sta=min_nb_sta,
            averaging=averaging,
        )
        res = res.loc[f'D{forecast_day}']
        stats = pd.concat(
            [stats, res.to_frame(name=lab).T],
            sort=False,
        )

    # print or save table
    if output_file is None:
        print(stats.to_string(float_format=float_format))
    else:
        with open(output_file, 'w') as f:
            if title is not None:
                f.write(title+'\n\n')
            f.write(stats.to_string(float_format=float_format))

    # save table with LaTeX layout
    if output_latex is not None:
        with open(output_latex, 'w') as f:
            if title is not None:
                f.write(title+'\n\n')
            tab = stats.rename(
                columns=lambda x: x.replace('_', ' '),
                index=lambda x: x.replace('_', ' '),
            )
            f.write(stats.to_latex(bold_rows=True, float_format=float_format))


def median_station_scores(
        objects, forecast_day, score_list, title=None,
        output_file=None, output_latex=None, labels=None,
        availability_ratio=0.75, min_nb_sta=10,
        float_format=None):
    """
    Build a table with median values of station scores.

    This function is based on Evaluator.stationScores method.
    Scores are first computed for each station at the wanted forecast day.
    Then, the median of these scores is taken.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for computation.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.
    score_list : list of str
        List of computed scores.
    title : str or None
        Title to add in the output file.
    output_file : str
        File where to save the table. If None, the table is printed.
    output_latex : str or None
        If not None, file where to save the table in a LaTeX layout.
    labels : None or list of str
        Labels of the objects.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        time to compute the scores for each station.
    min_nb_sta : int
        Minimal number of stations required to compute the median of
        the scores.
    float_format : str
        String format for table values.

    """
    return average_scores(
        objects=objects,
        forecast_day=forecast_day,
        score_list=score_list,
        title=title,
        output_file=output_file,
        output_latex=output_latex,
        labels=labels,
        availability_ratio=availability_ratio,
        min_nb_sta=min_nb_sta,
        float_format=float_format,
        averaging='median',
        score_type='temporal',
    )


def exceedances_scores(
        objects, forecast_day, thresholds, score_list=None,
        title=None, output_file=None, output_latex=None,
        labels=None, float_format=None, start_end=None):
    """
    Contingency table.

    Tables corresponding to the different thresholds are stored in the
    same file.
    Table values are computed with evaltools.scores.exceedances_scores
    function.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for computation.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.
    thresholds : list of scalar
        Threshold values.
    score_list : list of str
        List of scores to display. If None, all available scores are displayed.
        Available scores are

          - 'accuracy' : Accuracy
          - 'bias_score' : Bias score
          - 'success_ratio' : Success ratio
          - 'hit_rate' : probability of detection (Hit rate)
          - 'false_alarm_ratio' : false alarm ratio
          - 'prob_false_detect' : probability of false detection
          - 'threat_score' : Threat Score
          - 'equitable_ts' : Equitable Threat Score
          - 'peirce_ss' : Peirce Skill Score (Hanssen and Kuipers discriminant)
          - 'heidke_ss' : Heidke Skill Score
          - 'rousseau_ss' : Rousseau Skill Score
          - 'odds_ratio' : Odds Ratio
          - 'odds_ratio_ss' : Odds Ratio Skill Score

    title : str or None
        Title to add in the output file.
    output_file : str or None
        File where to save the tables. If None, the tables are printed.
    output_latex : str or None
        If not None, file where to save the tables in a LaTeX layout.
    labels : None or list of str
        Labels of the objects.
    float_format : str
        String format for table values.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    """
    # build 'labels' from objects 'model' attribute if 'labels' is None
    if labels is None:
        labels = [obj.model for obj in objects]

    # score_list default value
    if score_list is None:
        score_list = [
            'accuracy', 'bias_score', 'success_ratio', 'hit_rate',
            'false_alarm_ratio', 'prob_false_detect', 'threat_score',
            'equitable_ts', 'peirce_ss', 'heidke_ss', 'rousseau_ss',
            'odds_ratio', 'odds_ratio_ss',
        ]

    # compute stats
    tables = {}
    for thr in thresholds:
        res = pd.DataFrame()
        for obj, lab in zip(objects, labels):
            obs = obj.get_obs(forecast_day, start_end=start_end)
            obs = obs.values.flatten()
            sim = obj.get_sim(forecast_day, start_end=start_end)
            sim = sim.values.flatten()
            stats = evt.scores.exceedances_scores(obs, sim, thr=thr)[0]
            stats = {k: stats[k] if k in stats else np.nan for k in score_list}
            res = pd.concat([res, pd.DataFrame(stats, index=[lab])],
                            sort=False)
            tables[thr] = res

    # print or save tables
    if output_file is None:
        if title is not None:
            print(title+'\n')
        for thr in thresholds:
            print("threshold = {}".format(thr))
            print(tables[thr].to_string(float_format=float_format))
            print('\n')
    else:
        with open(output_file, 'w') as f:
            if title is not None:
                f.write(title+'\n\n')
            for thr in thresholds:
                f.write("threshold = {}\n".format(thr))
                f.write(tables[thr].to_string(float_format=float_format))
                f.write('\n\n')

    # save tables with LaTeX layout
    if output_latex is not None:
        with open(output_latex, 'w') as f:
            if title is not None:
                f.write(title+'\n\n')
            for thr in thresholds:
                f.write("threshold = {}\n".format(thr))
                tab = tables[thr].rename(
                    columns=lambda x: x.replace('_', ' '),
                    index=lambda x: x.replace('_', ' '),
                )
                f.write(
                    tab.to_latex(bold_rows=True, float_format=float_format)
                )
                f.write('\n\n')

    return [tables[thr] for thr in thresholds]
