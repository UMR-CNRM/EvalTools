# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers scores plotting functions."""

import datetime as dt
import re
from functools import reduce
from itertools import repeat
from copy import deepcopy

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, ttest_ind, mannwhitneyu
from scipy.stats import scoreatpercentile

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import evaltools as evt
import evaltools.translation as tr
from . import _mpl
from ._utils import plot_func, IMPLEMENTED_PLOTS

# matplotlib figure size (width, height in inches)
LANG = 'ENG'
USER_INTERFACE_WINDOW = True

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['legend.fontsize'] = 'small'
# plt.style.use('seaborn')
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams["figure.autolayout"] = True


def _check_period(objects, stop=True):
    """Check if Evaluator objects are defined on the same period."""
    starts = np.array([obj.start_date for obj in objects])
    ends = np.array([obj.end_date for obj in objects])
    if not (starts == starts[0]).all() or not (ends == ends[0]).all():
        if stop is False:
            print("Warning: All Evaluator objects are not defined " +
                  "on the same period !!!")
        else:
            raise evt.EvaltoolsError(
                "All Evaluator objects must be defined on the same period !!!")


def _check_station_list(objects):
    """Check if Evaluator objects use the same station list."""
    sta_union = reduce(
        np.union1d,
        [np.array(obj.stations.index) for obj in objects],
    )
    sta_inter = reduce(
        np.intersect1d,
        [np.array(obj.stations.index) for obj in objects],
    )
    if sta_union.shape[0] != sta_inter.shape[0]:
        print("Warning: All Evaluator objects do not use " +
              "the same station list !!!")


def _write_csv(data, output_csv, **kwargs):
    """
    Save DataFrame in csv format.

    data : panda.DataFrame
        Data to write.
    output_csv : str or None
        File where to save the data.
    **kwargs :
        Keys of these arguments must be contained in the name of the
        output file and will be replaced by the corresponding values.

    """
    # check compulsory keys in file name
    for key in kwargs:
        if "{" + key + "}" not in output_csv:
            raise evt.EvaltoolsError(
                "'{" + key + "}' must be "
                "contained in the name of the output csv file !!!"
            )

    data.to_csv(
        output_csv.format(**kwargs),
        na_rep='nan', float_format='%g', header=True, index=True,
    )


def _default_params(
        objects, labels, colors, linestyles, markers, default_marker=''):
    """Define default parameters in several plotting functions."""
    # check if all objects have the same series_type
    types = np.array([obj.series_type for obj in objects])
    series_type = types[0]
    if not (types == series_type).all():
        raise evt.EvaltoolsError(
            "All objects must have the same series_type attribute !!!")

    # define objects labels in case of None
    if labels is None:
        labels = []
        for obj in objects:
            labels.append(obj.model)

    # define objects colors in case of None
    if colors is None:
        colors = []
        for obj in objects:
            colors.append(obj.color)

    # define linestyles in case of None
    if linestyles is None:
        linestyles = ['-']*len(objects)

    # define markers in case of None
    if markers is None:
        markers = [default_marker]*len(objects)

    # choose nb of columns in legend
    if len(objects) > 4:
        legend_ncol = 2
    else:
        legend_ncol = 1

    return labels, colors, linestyles, markers, legend_ncol, series_type


def _find_any_missing(
        objects, labels, forecast_day, start_end=None, station_list=None):
    """
    Find indices where we have a missing value in a least one object.

    This function manages objects with different time steps or defined
    over different periods.

    Parameters
    ----------
    objects : List of evaltools.Evaluator objects
        Evaluator objects used for computation.
    labels : list of str
        List of labels corresponding to the objects.
    forecast_day : int
        Integer corresponding to the chosen forecast_day used for computation.
    start_end : None or list of two datetime.date objects
        Period to take into account.
    station_list : None or list of str
        List of stations to take into account. If None, all stations are
        considered.

    Returns
    -------
        Dictionary which keys are the labels and values are boolean arrays
        where positve values mean that one of the objects misses the
        corresponding value.

    """
    try:
        assert len(objects) == len(labels)
        assert len(labels) == len(np.unique(labels))
    except AssertionError:
        raise evt.EvaltoolsError(
            "Labels do not correspond to objects or are duplicated."
        )

    nan_idx = {lab: False for lab in labels}
    for lab, obj in zip(labels, objects):
        time_idx = evt.evaluator.Evaluator.get_sim(
                obj,
                forecast_day=forecast_day,
                start_end=start_end,
            ).index
        for o in objects:
            sim_df = evt.evaluator.Evaluator.get_sim(
                o,
                forecast_day=forecast_day,
                start_end=start_end,
            )
            if station_list is not None:
                sim_df = sim_df[station_list]
            sim_df = sim_df.reindex(time_idx, fill_value=1)
            nan_idx[lab] = np.logical_or(nan_idx[lab], np.isnan(sim_df))
        if isinstance(objects[0], evt.evaluator.Evaluator):
            obs_df = objects[0].get_obs(
                forecast_day=forecast_day,
                start_end=start_end,
            )
            if station_list is not None:
                obs_df = obs_df[station_list]
            obs_df = obs_df.reindex(time_idx, fill_value=1)
            nan_idx[lab] = np.logical_or(nan_idx[lab], np.isnan(obs_df))

    return nan_idx


def _process_data_values(objects, labels, obs_label, forecast_day=0):
    """
    Format concentration values to be plotted.

    Return a pandas.DataFrame containing concentration values for every
    objects and for observations without nan values.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects to be processed.
    labels : list of str
        List of labels coresponding to each object.
    obs_label : str
        Label given to observations.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.

    Returns
    -------
        pandas.DataFrame

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # find indices where we have no missing value in every objects
    nan_idx = _find_any_missing(
        objects,
        labels=labels,
        forecast_day=forecast_day,
    )

    # plotting
    data = pd.DataFrame()
    for obj, lab in zip(objects, labels):
        # compute scores
        sim = deepcopy(obj.get_sim(forecast_day=forecast_day))
        sim[nan_idx[lab]] = np.nan
        data[lab] = sim.values.flatten()

    # obs are plotted only for the first object in the list
    obs = deepcopy(objects[0].get_obs(forecast_day=forecast_day))
    obs = obs.where(~nan_idx[labels[0]], np.nan)
    data[obs_label] = obs.values.flatten()

    return data


def _process_temporal_score(
        objects, score, labels, nb_stations=False,
        availability_ratio=.75, forecast_day=0):
    """
    Compute a score for each station.

    Return a pandas.DataFrame containing score values for every
    objects.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects to be processed.
    score : str
        Computed score.
    labels : list of str
        List of labels coresponding to each object.
    nb_stations : bool
        If True, the number of stations (with enough data availability) is
        added to the labels.
    availability_ratio : float
        Minimal rate of data availability required on the period to compute
        the scores.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.

    Returns
    -------
        pandas.DataFrame

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    data = pd.DataFrame()
    for obj, lab in zip(objects, labels):
        res = obj.temporal_scores(
            [score],
            availability_ratio=availability_ratio,
        )
        col = (
            lab + f" ({sum(res[f'D{forecast_day}'][score].notna())} stations)"
            if nb_stations
            else lab
        )
        data[col] = res[f'D{forecast_day}'][score]

    return data


def _process_spatial_score(
        objects, score, labels, min_nb_sta=10, forecast_day=0):
    """
    Compute spatial scores for each Evaluator object.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects to be processed.
    score : str
        Computed score.
    labels : list of str
        List of labels coresponding to each object.
    nb_stations : bool
        If True, the number of stations (with enough data availability) is
        added to the labels.
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute the scores.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.

    Returns
    -------
        pandas.DataFrame

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    data = pd.DataFrame()
    for obj, lab in zip(objects, labels):
        res = obj.spatial_scores(
            [score],
            min_nb_sta=min_nb_sta,
        )
        data[lab] = res[f'D{forecast_day}'][score]

    return data


def _compute_kde(data, bw_method=None):
    """
    Compute probability density function.

    For each columns of the input DataFrame, the probability density function
    is computed with scipy.stats.gaussian_kde.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to process.
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be
        ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar,
        this will be used directly as kde.factor. If a callable, it should
        take a gaussian_kde instance as only parameter and return a scalar.
        If None (default), ‘scott’ is used.

    Returns
    -------
        pandas.DataFrame

    """
    x_min = np.nanmin(data.values)
    x_max = np.nanmax(data.values)
    x = np.linspace(x_min, x_max, 1000)
    res = {}
    for col in data:
        idx_nan = np.isnan(data[col].values)
        try:
            kernel = gaussian_kde(
                data[col].values[~idx_nan],
                bw_method=bw_method,
            )
            res[col] = kernel(x)
        except Exception as err:
            print(type(err), err)
            print(
                f"Warning: KDE for {col} can't be computed (data "
                "may contain Inf values)"
            )

    res = pd.DataFrame(res, index=x)

    return res


@plot_func
def plot_data_density(
        objects, forecast_day=0, labels=None, colors=None,
        linestyles=None, title="", xmin=None, xmax=None,
        obs_style=None, black_axes=False,
        fig=None, ax=None):
    """
    Plot the probability density function.

    Draw the probability density of observed (for the first object only)
    and simulated data. This function is based on kernel density estimation.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    title : str
        Title for the plot.
    xmin, xmax : None or scalar
        Limits of the x axis.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.

    """
    # define default parameters
    markers = None
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    if not obs_style:
        obs_style = {'color': 'k', 'alpha': 0.5}
    else:
        obs_style = deepcopy(obs_style)

    obs_label = obs_style.pop('label', 'observations')

    data = _process_data_values(
        objects,
        labels=labels,
        obs_label=obs_label,
        forecast_day=forecast_day,
    )
    data = _compute_kde(data)
    fig, ax = _mpl.mpl_plot_line(
        data.drop(obs_label, axis=1),
        plot_kw=dict(
            color=colors,
            style=linestyles,
            # style=[l+m for m, l in zip(markers, linestyles)],
        ),
        legend=False,
        fig=fig, ax=ax,
    )
    fig, ax = _mpl.mpl_plot_line(
        data[obs_label],
        plot_kw=obs_style,
        legend_kw=dict(ncol=legend_ncol),
        legend_type='ax',
        fig=fig,
        ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title=title,
        ylabel=tr.station_score_density['ylabel'][LANG],
        black_axes=black_axes,
        xmin=xmin,
        xmax=xmax,
    )

    return fig, ax


@plot_func
def plot_data_box(
        objects, forecast_day=0, labels=None, colors=None,
        obs_style={}, title="", showfliers=True,
        fig=None, ax=None):
    """
    Plot distribution of the data values in a boxplot.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k'}.
    title : str
        Title for the plot.
    showfliers : bool
        Show the outliers beyond the caps.

    """
    # define default parameters
    markers = None
    linestyles = None
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    obs_label = obs_style.get('label', 'observations')
    obs_color = obs_style.get('color', 'k')

    data = _process_data_values(
        objects,
        labels=labels,
        obs_label=obs_label,
        forecast_day=forecast_day,
    )

    fig, ax = _mpl.mpl_plot_box(
        data,
        colors=colors + [obs_color],
        plot_kw=dict(vert=False, showfliers=showfliers),
        fig=fig,
        ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title=title,
    )

    return fig, ax


@plot_func
def plot_score_density(
        objects, score, forecast_day=0,
        score_type='temporal', availability_ratio=0.75, min_nb_sta=10,
        labels=None, colors=None, linestyles=None,
        title="", nb_stations=False,
        fig=None, ax=None):
    """
    Plot the probability density function of score values.

    This function is based on kernel density estimation.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    score_type : str
        Computing method selected among 'temporal' or 'spatial'.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute a temporal score (ignored if score_type =
        'spatial').
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute a spatial score (ignored if
        score_type = 'temporal').
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    title : str
        Title for the plot.
    nb_stations : bool
        If True, write the number of stations in the legend (ignored if
        score_type = 'spatial').

    """
    # define default parameters
    markers = None
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    if score_type == 'temporal':
        res = _process_temporal_score(
            objects,
            score=score,
            labels=labels,
            availability_ratio=availability_ratio,
            forecast_day=forecast_day,
            nb_stations=nb_stations,
        )
    elif score_type == 'spatial':
        res = _process_spatial_score(
            objects,
            score=score,
            labels=labels,
            min_nb_sta=min_nb_sta,
            forecast_day=forecast_day,
        )
    else:
        raise evt.EvaltoolsError(
            "score_type argument must be either equal to "
            "'temporal' or 'spatial'."
        )

    data = _compute_kde(res)

    fig, ax = _mpl.mpl_plot_line(
        data,
        plot_kw=dict(
            color=colors,
            style=linestyles,
        ),
        legend=True,
        legend_type='ax',
        fig=fig, ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title=title,
    )

    return fig, ax


def plot_station_score_density(
        objects, score, forecast_day=0, availability_ratio=0.75,
        labels=None, colors=None, linestyles=None,
        title="", nb_stations=False, annotation=None,
        output_file=None, file_formats=['png'],
        fig=None, ax=None):
    """
    Plot the probability density function of score values.

    This function is based on kernel density estimation.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute a temporal score.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    title : str
        Title for the plot.
    nb_stations : bool
        If True, write the number of stations in the legend.
    annotation : str or None
        Additional information to write in the upper left corner of the plot.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    return plot_score_density(
        objects=objects, score=score, forecast_day=forecast_day,
        score_type='temporal', availability_ratio=availability_ratio,
        labels=labels, colors=colors, linestyles=linestyles,
        title=title, nb_stations=nb_stations, annotation=annotation,
        output_file=output_file, file_formats=file_formats,
        fig=fig, ax=ax,
    )


@plot_func
def plot_score_box(
        objects, score, forecast_day=0,
        score_type='temporal', availability_ratio=0.75, min_nb_sta=10,
        labels=None, colors=None, title="", nb_stations=False, showfliers=True,
        fig=None, ax=None):
    """
    Plot distribution of the score values in a boxplot.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    score_type : str
        Computing method selected among 'temporal' or 'spatial'.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute a temporal score (ignored if score_type =
        'spatial').
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute a spatial score (ignored if
        score_type = 'temporal').
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    title : str
        Title for the plot.
    nb_stations : bool
        If True, the number of stations used to draw the boxes is
        displayed in the legend (ignored if score_type = 'spatial').
    showfliers : bool
        Show the outliers beyond the caps.

    """
    # define default parameters
    markers = None
    linestyles = None
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    if score_type == 'temporal':
        data = _process_temporal_score(
            objects,
            score=score,
            labels=labels,
            availability_ratio=availability_ratio,
            forecast_day=forecast_day,
            nb_stations=nb_stations,
        )
    elif score_type == 'spatial':
        data = _process_spatial_score(
            objects,
            score=score,
            labels=labels,
            min_nb_sta=min_nb_sta,
            forecast_day=forecast_day,
        )
    else:
        raise evt.EvaltoolsError(
            "score_type argument must be either equal to "
            "'temporal' or 'spatial'."
        )

    fig, ax = _mpl.mpl_plot_box(
        data,
        colors=colors,
        plot_kw=dict(vert=False, showfliers=showfliers),
        fig=fig,
        ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title=title,
        # xticks_kw={},
    )

    return fig, ax


def plot_station_score_box(
        objects, score, forecast_day=0, availability_ratio=0.75,
        labels=None, colors=None, title="", nb_stations=False, showfliers=True,
        annotation=None, output_file=None, file_formats=['png'],
        fig=None, ax=None):
    """
    Plot distribution of the score values in a boxplot.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute a temporal score.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    title : str
        Title for the plot.
    nb_stations : bool
        If True, the number of stations used to draw the boxes is
        displayed in the legend.
    showfliers : bool
        Show the outliers beyond the caps.
    annotation : str or None
        Additional information to write in the upper left corner of the plot.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    return plot_score_box(
        objects=objects, score=score, forecast_day=forecast_day,
        score_type='temporal', availability_ratio=availability_ratio,
        labels=labels, colors=colors, title="", nb_stations=nb_stations,
        showfliers=showfliers, annotation=annotation, output_file=output_file,
        file_formats=file_formats, fig=fig, ax=ax,
    )


def plot_mean_time_scores(
        objects, score, score_name=None,
        min_nb_sta=10, availability_ratio=0.75,
        labels=None, colors=None, linestyles=None, markers=None,
        title="", xlabel=None, black_axes=False,
        nb_of_minor_ticks=(3, 1),
        outlier_thresh=None,
        output_csv=None, annotation=None,
        output_file=None, file_formats=['png'], fig=None, ax=None):
    """
    Plot the temporal mean of spatial scores for each forecast time.

    This function is based on Evaluator.average_ft_scores method.
    The score is first computed for each time along available stations. Then,
    the mean of this score is taken for each forecast time.

    On the plot, there will be as many lines as there are objects given in
    the object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    score_name : str | dict | None
        String to write in the title instead of the default score name.
        If a dictionary is passed, it must have a key corresponding to the
        score argument. If None, the score name is written as in the score
        argument.
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute the scores (before applying
        mean for each forecast time).
    availability_ratio : float
        Minimal rate of data (computed scores per time) available on the
        period required per forecast time to compute the mean scores.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plot. It can contain {score} instead of the
        score name.
    xlabel : str
        Label for the x axis.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    output_csv : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.
    annotation : str or None
        Additional information to write in the upper left corner of the plot.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    return plot_average_ft_scores(
        objects=objects,
        score=score,
        score_name=score_name,
        min_nb_sta=min_nb_sta,
        availability_ratio=availability_ratio,
        score_type='spatial',
        averaging='mean',
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        markers=markers,
        title=title,
        xlabel=xlabel,
        black_axes=black_axes,
        nb_of_minor_ticks=nb_of_minor_ticks,
        outlier_thresh=outlier_thresh,
        annotation=annotation,
        output_file=output_file,
        file_formats=file_formats,
        output_csv=output_csv,
        fig=fig,
        ax=ax,
    )


def plot_median_station_scores(
        objects, score, score_name=None,
        min_nb_sta=10, availability_ratio=0.75,
        labels=None, colors=None, linestyles=None, markers=None,
        title="", xlabel=None, black_axes=False,
        nb_of_minor_ticks=(3, 1),
        outlier_thresh=None, annotation=None,
        output_file=None, file_formats=['png'],
        output_csv=None, fig=None, ax=None):
    """
    Plot the spatial median of temporal scores for each forecast time.

    This function is based on Evaluator.average_ft_scores method.
    The score is first computed for each station at every times. Then,
    the median of these score values is taken for each forecast time.

    On the plot, there will be as many lines as there are objects given in
    the object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    score_name : str | dict | None
        String to write in the title instead of the default score name.
        If a dictionary is passed, it must have a key corresponding to the
        score argument. If None, the score name is written as in the score
        argument.
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute the scores (before applying
        mean for each forecast time).
    availability_ratio : float
        Minimal rate of data (computed scores per time) available on the
        period required per forecast time to compute the mean scores.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plot. It can contain {score} instead of the
        score name.
    xlabel : str
        Label for the x axis.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    output_csv : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.
    annotation : str or None
        Additional information to write in the upper left corner of the plot.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    return plot_average_ft_scores(
        objects=objects,
        score=score,
        score_name=score_name,
        min_nb_sta=min_nb_sta,
        availability_ratio=availability_ratio,
        score_type='temporal',
        averaging='median',
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        markers=markers,
        title=title,
        xlabel=xlabel,
        black_axes=black_axes,
        nb_of_minor_ticks=nb_of_minor_ticks,
        outlier_thresh=outlier_thresh,
        annotation=annotation,
        output_file=output_file,
        file_formats=file_formats,
        output_csv=output_csv,
        fig=fig,
        ax=ax,
    )


@plot_func
def plot_average_ft_scores(
        objects, score, score_name=None,
        min_nb_sta=10, availability_ratio=0.75,
        score_type='temporal', averaging='median',
        labels=None, colors=None, linestyles=None, markers=None,
        title="", xlabel=None, black_axes=False,
        nb_of_minor_ticks=(3, 1),
        outlier_thresh=None,
        output_csv=None, fig=None, ax=None):
    """
    Plot the average score for each forecast time.

    This function is based on Evaluator.average_ft_scores method.
    If score_type is 'temporal', the score is first computed for each
    station at every times. Then, the spatial average of these score values
    is taken for each forecast time.
    If score_type is 'spatial', the score is first computed for each time and
    then the temporal average of these score values is taken for each forecast
    time.

    On the plot, there will be as many lines as there are objects given in
    the object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    score_name : str | dict | None
        String to write in the title instead of the default score name.
        If a dictionary is passed, it must have a key corresponding to the
        score argument. If None, the score name is written as in the score
        argument.
    min_nb_sta : int
        Minimal number of stations required to compute the average of
        the score if score_type is 'temporal, or to compute the score itself
        if score_type is 'spatial'.
    availability_ratio : float
        Minimum required rate of available data over the period to
        calculate the score if score_type is 'temporal, or to calculate
        the average of the score if score_type is 'spatial'.
    score_type : str
        Computing method selected from 'temporal' or 'spatial'.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plot. It can contain {score} instead of the
        score name.
    xlabel : str
        Label for the x axis.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    output_csv : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    # define default score names
    if isinstance(score_name, dict):
        score_name = score_name[score]
    if score_name is None:
        score_name = score
    if xlabel is None:
        xlabel = tr.mean_time_scores['xlabel'][LANG]

    # abcissa
    if objects[0].series_type == 'hourly':
        x_max = 24*objects[0].forecast_horizon
        x = range(0, x_max, objects[0].step)
        major_ticks = np.arange(0, x_max, 12)
        # minor_ticks = np.arange(0, x_max, 3)
        xticks_kw = dict(ticks=major_ticks)
    else:
        x_max = objects[0].forecast_horizon
        x = range(0, x_max)
        xticks_kw = dict(ticks=x, labels=[f'D{fd}' for fd in x])

    # scores computing
    stats = pd.DataFrame()
    for obj, lab in zip(objects, labels):
        stats[lab] = obj.average_ft_scores(
            [score],
            min_nb_sta=min_nb_sta,
            availability_ratio=availability_ratio,
            score_type=score_type,
            averaging=averaging,
        )[score]

    fig, ax = _mpl.mpl_plot_line(
        stats,
        plot_kw=dict(
            color=colors,
            style=[l+m for m, l in zip(markers, linestyles)],
            # linestyle=linestyles,
            # marker=markers,
        ),
        legend_type='ax',
        legend_kw=dict(ncol=(2 if len(objects) > 4 else 1)),
        outlier_thresh=outlier_thresh,
        fig=fig, ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        xlabel=xlabel,
        ylabel=score_name,
        black_axes=black_axes,
        title=title.format(score=score_name),
        xticks_kw=xticks_kw,
        xmin=x[0],
        xmax=x[-1],
        nb_of_minor_ticks=nb_of_minor_ticks,
    )

    # try to control number of yticks
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)

    # save data
    if output_csv is not None:
        stats.to_csv(output_csv.format(score=score))

    return fig, ax


def plot_station_scores(
        obj, score, ref=None, forecast_day=0, output_file=None, title="",
        bbox=None, file_formats=['png'], point_size=5,
        higher_above=True, order_by=None, availability_ratio=0.75,
        vmin=None, vmax=None, vcenter=None, cmap=None, norm=None,
        rivers=False, output_csv=None,
        interp2d=False, sea_mask=False, land_mask=False, bnd_resolution='50m',
        cmaplabel='', extend='neither', land_color='none', sea_color='none',
        marker='o', mark_by=None, boundaries_above=False, bbox_inches='tight',
        grid_resolution=None, ne_features=None, fig=None, ax=None):
    """
    Plot scores per station on a map.

    This function is based on Evaluator.stationScores.

    Each station is drawn with a circle colored according to
    it score value.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Object used for plotting.
    score : str
        Computed score.
    ref : None or evaltools.Evaluator object
        Reference object used for comparison. If provided, values plotted on
        the map will be de score difference between the main object and the
        reference object. For scores where the better is to be close to zero
        (like mean bias), the difference is calculated with absolute values.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    title : str
        Title for the plot.
    bbox : list of floats
        Bounding box of plotting area [min_lon, max_lon, min_lat, max_lat].
    file_formats : list of str
        List of file extensions.
    point_size : float
        Point size (as define in matplotlib.pyplot.scatter).
    higher_above : bool
        If True, stations with higher score are plotted above. If False,
        stations with lower score are plotted above.
    order_by : str
        Vertically order point according to this argument. It must be a
        column of the Evaluator object stations attribute.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute the mean scores.
    vmin, vmax : None or scalar
        Min and max values for the legend colorbar. If None, the respective
        min and max of the score values are used.
    vcenter : None or scalar
        If not None, matplotlib.colors.TwoSlopeNorm(vcenter, vmin, vmax) is
        used as the colorbar norm.
    cmap : matplotlib.colors.Colormap object
        Colors used for plotting (default: matplotlib.cm.jet).
    norm : matplotlib.colors.Normalize
        The Normalize instance scales the data values to the canonical
        colormap range [0, 1] for mapping to colors.
    rivers : bool
        If True, rivers and lakes are drawn.
    output_csv : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.
    interp2d : bool
        If True, a 2D linear interplation is performed on scores values.
    sea_mask : bool
        If True, scores ought to be drawn over sea are masked.
    land_mask : bool
        If True, scores ought to be drawn over land are masked.
    bnd_resolution : str
        Resolution of coastlines and boundary lines. It can be '10m',
        '50m' or '110m'.
    cmaplabel : str
        Label for the colormap.
    extend : str
        Chosen among 'neither', 'both', 'min' or 'max'. If not 'neither', make
        pointed end(s) to colorbar for out-of-range values.
    land_color, sea_color : str
        Land/sea colors.
    marker : str
        The marker style (ignored if you pass a mark_by instance).
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of self.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    boundaries_above : bool
        If True, boundaries and coast lines are drawn above score data.
    bbox_inches : str or matplotlib.transforms.Bbox
        Bounding box in inches: only the given portion of the figure is saved.
        If 'tight', try to figure out the tight bbox of the figure.
    grid_resolution : couple of scalars
        Couple of scalars corresponding to meridians and parallels spacing
        in degrees.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or cartopy.mpl.geoaxes.GeoAxes
        Axis to use for the plot. If None, a new axis is created.
    ne_features : list of dicts
        Each dictionary contains arguments to instanciate a
        cartopy.feature.NaturalEarthFeature(...).
        E.g. [dict(category='cultural', name='admin_1_states_provinces',
        facecolor='none', linestyle=':'),] will add
        states/departments/provinces. If this argument is provided,
        rivers, land_color and sea_color arguments will not be taken into
        account.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    cartopy.mpl.geoaxes.GeoAxes
        Axes object of the produced plot.

    """
    import cartopy.crs as ccrs

    if not np.isin(['lat', 'lon'], obj.stations.columns).all():
        print(
            f"lat/lon coordinates not found in {obj}.stations. Station "
            "scores map can not be plotted."
        )
        return None

    if bbox:
        obj = obj.sub_area(*bbox, inplace=False)

    # compute scores
    df = obj.temporal_scores(
        [score],
        availability_ratio=availability_ratio,
    )[f'D{forecast_day}']

    cmap = cmap or cm.jet
    if vcenter is not None:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vcenter=vcenter, vmax=vmax, vmin=vmin)
    elif vmin is not None or vmax is not None:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)

    if ref:
        df_ref = ref.temporal_scores(
            [score],
            availability_ratio=availability_ratio,
        )[f'D{forecast_day}']
        stations = np.intersect1d(
            obj.stations.index,
            ref.stations.index,
        )
        score_obj = evt.scores.get_score(score)
        if score_obj.sort_key == evt.scores._sort_keys.absolute:
            df = df.loc[stations].abs() - df_ref.loc[stations].abs()
        else:
            df = df.loc[stations] - df_ref.loc[stations]

    # sort values by plotted values (or metadata if order_by is not None)
    df.sort_values(by=score, ascending=higher_above, inplace=True)
    if order_by is not None:
        if np.in1d(order_by, obj.stations.columns):
            df = df.merge(
                obj.stations[[order_by]], how='left',
                left_index=True, right_index=True, sort=False,
            )
            df.sort_values(
                ascending=higher_above, by=order_by, inplace=True,
            )
        else:
            print(
                f"*** Values can't be ordered by {order_by} since "
                "this variable is not in the object's "
                "metadata."
            )
    df = df[score]

    # save data
    if output_csv is not None:
        csv_data = obj.stations[['lat', 'lon']].loc[df.index]
        csv_data[score] = df
        csv_data.index.name = 'station'
        csv_data.sort_values(by=score)
        csv_data.to_csv(
            output_csv.format(score=score), sep=' ',
            na_rep='nan', float_format='%g', header=True,
            index=True,
        )

    # plotting
    fig, ax = _mpl.set_cartopy(
        fig=fig,
        ax=ax,
        grid_resolution=grid_resolution,
        bbox=bbox,
        rivers=rivers,
        land_color=land_color,
        sea_color=sea_color,
        bnd_resolution=bnd_resolution,
        boundaries_above=boundaries_above,
        sea_mask=sea_mask,
        land_mask=land_mask,
        ne_features=ne_features,
    )

    # title
    plt.title(title)

    # annotate number of stations
    annotation = tr.station_scores['annotation'][LANG].format(
        av=sum(df.notna()),
        total=df.shape[0],
        mini=round(df.min(), 2),
        avg=round(df.mean(), 2),
        maxi=round(df.max(), 2)
    )
    plt.annotate(
        annotation,
        xy=(0, 0),
        xytext=(0, -10),
        xycoords='axes fraction',
        textcoords='offset points',
        va='top',
        fontsize='small',
        fontstyle='italic',
        color='#7A7A7A',
    )

    if interp2d is True:
        from scipy.interpolate import griddata
        x1, x2, y1, y2 = bbox
        grid_lat, grid_lon = np.mgrid[y1:y2:1000j, x1:x2:1000j]
        interp_data = griddata(
            np.array(obj.stations[['lat', 'lon']].loc[df.index]),
            df.values, (grid_lat, grid_lon),
            method='linear')
        art = ax.imshow(
            interp_data, extent=bbox,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            zorder=2,
            origin='lower',
        )
        # art = ax.pcolormesh(grid_lon[0, :], grid_lat[:, 0], interp_data,
        #                    transform=ccrs.Mercator())

    else:
        # define scatter plot markers
        if mark_by is None:
            markers = None
        else:
            df.dropna(inplace=True)
            markers = [mark_by[1][obj.stations[mark_by[0]][code]]
                       for code in df.index]
            handles = [mlines.Line2D([], [], color='grey',
                                     marker=mark_by[1][key],
                                     label=key, linestyle='')
                       for key in mark_by[1]]
            ax.legend(handles=handles)
            ax.legend_.set_zorder(12)
        # scatter plot
        x = np.array(obj.stations.lon.loc[df.index])
        y = np.array(obj.stations.lat.loc[df.index])
        art = _mpl.mscatter(
            x, y, ax=ax, m=markers, marker=marker, s=point_size, c=df,
            cmap=cmap, edgecolors=None, norm=norm,
            transform=ccrs.PlateCarree(), zorder=2,
        )

    # custom axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="3%",
        axes_class=plt.Axes,
        pad=0.2,
    )
    plt.colorbar(art, cax=cax, extend=extend, label=cmaplabel)

    # save figure
    _mpl.save_figure(output_file, file_formats, bbox_inches=bbox_inches)

    return fig, ax


IMPLEMENTED_PLOTS['station_scores'] = plot_station_scores


@plot_func
def plot_time_scores(
        objects, score, term, hourly_timeseries=False, min_nb_sta=10,
        labels=None, colors=None, linestyles=None, markers=None,
        start_end=None, title="", score_name=None, black_axes=False,
        nb_of_minor_ticks=(2, 2), xticking='auto', date_format=None,
        outlier_thresh=None,
        output_csv=None, fig=None, ax=None):
    """
    Plot time scores at a chosen time for each forecast day.

    This function is based on Evaluator.timeScores method.
    On each plots, there will be as many lines as there are objects the
    object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    term : int.
        Integer corresponding to the chosen term used for plotting.
        If the series type of the objects is 'hourly', it refers
        to the forecast time (for example between 0 and 95 if the forecast
        horizon is 4 days). If the series type of the objects is 'daily',
        it refers to the forcast day (for example between 0 and 3 if the
        forecast horizon is 4 days).
    hourly_timeseries : bool
        If True, every time step are plotted. In this case, argument term
        refers to the forecast day. This argument is ignored if the series
        type of the objects is 'daily'.
    min_nb_sta : int
        Minimal number of station values available in both obs and sim
        required per datetime to compute the scores.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plot. It can contain {score} instead of the
        score name.
    score_name : str | dict | None
        String to write in the title instead of the default score name.
        If a dictionary is passed, it must have a key corresponding to the
        score argument. If None, the score name is written as in the score
        argument.
    xticking : str
        Defines the method used to set x ticks. It can be 'auto' (automatic),
        'mondays' (a tick every monday) or 'daily' (a tick everyday).
    date_format : str
        String format for dates as understood by python module datetime.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    start_end : None or list of two datetime.date objects
        Boundary dates for abscissa.
    output_csv : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)

    # define whether if term is the forecast day or forecast time
    if series_type == 'hourly':
        if hourly_timeseries:
            day = term
        else:
            # forecast day corresponding to term
            day = term//24
            # hour of the day corresponding term
            t = term % 24
    elif series_type == 'daily':
        day = term

    # define default score names
    if isinstance(score_name, dict):
        score_name = score_name[score]
    if score_name is None:
        score_name = score

    # scores computing
    stats = pd.DataFrame()
    for obj, lab in zip(objects, labels):
        st = obj.spatial_scores(
            [score],
            min_nb_sta=min_nb_sta,
        )
        st = st[f'D{day}']
        if st[score].isna().all():
            print(f"*** all nan values for {score} of {lab}")
        if series_type == 'hourly' and not hourly_timeseries:
            idx = st.index.time == dt.time(t, 0)
            st = st[idx]
        stats[lab] = st[score]

    # set xticks
    if start_end is None:
        # start_end = (xticks[0], xticks[-1])
        start_end = (stats.index[0], stats.index[-1])

    # for easier x axis formatting
    stats.index.freq = None

    fig, ax = _mpl.mpl_plot_line(
        stats,
        plot_with_pandas=False,
        plot_kw=dict(
            color=colors,
            # style=[l+m for m, l in zip(markers, linestyles)],
            linestyle=linestyles,
            marker=markers,
        ),
        legend_type='ax',
        legend_kw=dict(ncol=(2 if len(objects) > 4 else 1)),
        outlier_thresh=outlier_thresh,
        fig=fig, ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        ylabel=score_name,
        black_axes=black_axes,
        title=title.format(score=score_name),
        xmin=start_end[0],
        xmax=start_end[1],
        nb_of_minor_ticks=nb_of_minor_ticks,
    )

    _mpl.set_time_xaxis(fig, ax, xticking, date_format=date_format)

    # try to control number of yticks
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)

    # save data
    if output_csv is not None:
        stats.to_csv(output_csv)

    return fig, ax


@plot_func
def plot_quarterly_score(
        files, labels, score, first_quarter=None, last_quarter=None,
        colors=None, linestyles=None, markers=None, title=None,
        thres=None, ylabel=None, origin_zero=False, black_axes=False,
        fig=None, ax=None):
    """
    Plot quarterly score values saved in some files.

    This function is based on Evaluator.quarterlyMedianScore method.
    On each plots, there will be as many lines as there are file paths
    in <files> argument.

    Parameters
    ----------
    files : list of str
        Paths of files used for plotting (one file corresponds to one line
        on the plot).
    labels : list of str
        List of labels for the legend (length of the label list must be
        equal to the length of the file list).
    score : str
        The score that should correspond to data stored in the files.
    first_quarter : str
        String corresponding to the oldest plotted quarter.
    last_quarter : str
        String corresponding to the latest plotted quarter.
    colors : list of str
        Line colors corresponding to each files.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plots.
    thres : None or float
        If not None, a horizontal yellow line is plotted with
        equation y = thres.
    ylabel : str
        Ordinate axis label.
    origin_zero : bool
        If True, minimal value of y axis is set to zero.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.

    """
    if len(files) > 4:
        legend_ncol = 2
    else:
        legend_ncol = 1

    colors = colors or repeat(None)
    linestyles = linestyles or ['-']
    markers = markers or ['.']

    # find first and last quarters in case of None
    if last_quarter is None or first_quarter is None:
        dfs = [pd.read_csv(f, sep=' ', index_col=0) for f in files]
        quarter_list = []
        for df in dfs:
            quarter_list.extend(
                [evt.quarter.Quarter.from_string(q) for q in df.index]
            )
        quarter_list.sort()
        first_quarter = first_quarter or quarter_list[0].string
        last_quarter = last_quarter or quarter_list[-1].string

    end_quarter = evt.quarter.Quarter.from_string(last_quarter)
    start_quarter = evt.quarter.Quarter.from_string(first_quarter)
    all_quarters = end_quarter.range(start_quarter)

    data = pd.DataFrame(index=[q.__repr__() for q in all_quarters])
    for kpi_file, lab in zip(files, labels):
        df = pd.read_csv(kpi_file, sep=' ', index_col=0)
        y_vals = pd.DataFrame(
            index=data.index,
            columns=[score],
        )
        y_vals.update(df)
        data[lab] = y_vals.astype('float')

    x = range(len(all_quarters))

    fig, ax = _mpl.mpl_plot_line(
        data,
        plot_with_pandas=False,
        plot_kw=dict(
            color=colors,
            linestyle=linestyles,
            marker=markers,
        ),
        legend_kw=dict(ncol=legend_ncol),
        legend_type='fig',
        fig=fig, ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        ylabel=ylabel or score,
        black_axes=black_axes,
        title=title,
        xticks_kw=dict(rotation=30, ha='right'),
        xmin=x[0],
        xmax=x[-1],
        ymin=(0 if origin_zero else None),
    )

    if thres is not None:
        ax.plot(x, [thres]*len(x), color='#FFA600')

    ax.locator_params(axis='y', nbins=10, min_n_ticks=10)

    return fig, ax


@plot_func
def plot_taylor_diagram(
        objects, forecast_day=0, norm=True, colors=None, markers=None,
        point_size=100, labels=None, title="",
        threshold=0.75, frame=False, crmse_levels=10,
        output_csv=None, fig=None, ax=None):
    """
    Taylor diagram.

    This function is based on Evaluator.spatiotemporal_scores method.
    Pearson correlation and variance ratio are first computed from all data
    of a choosen forecast day (values for all station at all times are
    considered as a simple 1D array).

    References
    ----------
        Karl E. Taylor (2001), "Summarizing multiple aspects of model
        performance in a single diagram", JOURNAL OF GEOPHYSICAL RESEARCH,
        VOL. 106.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecast_day : int
        Forecast day for which to use data to compute the Taylor diagram.
    norm : bool
        If True, standard deviation and CRMSE are divided by the standard
        deviation of observations.
    colors : None or list of str
        Marker colors corresponding to each object.
    markers : None or list of str
        Marker shapes corresponding to each object.
    point_size : float
        Point size (as define in matplotlib.pyplot.scatter).
    labels : list of str
        List of labels for the legend.
    title : str
        Diagram title.
    threshold : int or float
        Minimal number (if type(threshold) is int) or minimal rate
        (if type(threshold) is float) of data available in both obs and
        sim required to compute the scores.
    frame : bool
        If false, top and right figure boundaries are not drawn.
    crmse_levels : int
        Number of CRMSE arcs of circle.
    output_csv : str or None
        File where to save the data.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # define default parameters
    linestyles = None
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(
            objects, labels, colors, linestyles, markers, default_marker='^',
        )
    # differences between original diagram and normalized one
    if norm:
        sd_label = tr.taylor_diagram['sdr'][LANG]
        score = 'SDratio'
        refstd = 1
    else:
        sd_label = tr.taylor_diagram['sd'][LANG]
        score = 'sim_std'
        refstd = objects[0].spatiotemporal_scores(
            score_list=['obs_std'],
            forecast_days=[forecast_day],
            threshold=threshold,
        ).iloc[0, 0]

    # compute scores
    csv_data = pd.DataFrame(columns=['corr', score, 'CRMSE'], dtype=float)
    x = []
    y = []
    for obj, lab in zip(objects, labels):
        df = obj.spatiotemporal_scores(
            score_list=[score, 'PearsonR', 'CRMSE'],
            forecast_days=[forecast_day],
            threshold=threshold,
        )
        corr = df['PearsonR'].loc[f'D{forecast_day}']
        score_value = df[score].loc[f'D{forecast_day}']
        x.append(corr*score_value)
        y.append(np.sin(np.arccos(corr))*score_value)
        csv_data = pd.concat(
            [
                csv_data,
                pd.DataFrame(
                    {
                        'corr': corr,
                        score: score_value,
                        'CRMSE': df['CRMSE'].loc[f'D{forecast_day}']
                    },
                    index=[lab],
                    dtype=float,
                )
            ],
            sort=False,
        )

    fig, ax = _mpl.mpl_taylor(
        x, y,
        labels=labels,
        colors=colors,
        markers=markers,
        scatter_kw=dict(s=point_size),
        refstd=refstd,
        crmse_levels=crmse_levels,
        crmse_labels=(not norm),
        # legend_kw=dict(loc='center left', bbox_to_anchor=(1, 0.5)),
        legend_type='ax',
        frame=frame,
    )

    _mpl.set_axis_elements(
        ax,
        xlabel=sd_label,
        ylabel=sd_label,
        title=title,
    )

    # save data
    if output_csv is not None:
        csv_data.index.name = 'model'
        csv_data.to_csv(
            output_csv, sep=' ', na_rep='nan', float_format='%g',
            header=True, index=True,
        )

    return fig, ax


@plot_func
def plot_score_quartiles(
        objects, xscore, yscore,
        score_type='temporal', forecast_day=0, availability_ratio=0.75,
        min_nb_sta=10, title="", colors=None, labels=None,
        invert_xaxis=False, invert_yaxis=False, black_axes=False,
        xmin=None, xmax=None, ymin=None, ymax=None,
        output_csv=None, fig=None, ax=None):
    """
    Scatter plot of median station scores.

    Plot the median of the station scores surrounded by a rectangle
    correponding to the first and third quartiles. This chart is
    based on Evaluator.stationScores method.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    xscore : str
        Score used for the x axis.
    yscore : str
        Score used for the y axis.
    score_type : str
        Computing method selected from 'temporal' or 'spatial'.
    forecast_day : int
        Forecast day used for score computing.
    availability_ratio : float
        Minimum required rate of available data over the period to
        calculate the scores (or the quartiles of the scores if
        score_type='spatial').
    min_nb_sta : int
        Minimal number of stations required to compute the quartiles of
        the scores (or the scores themself if score_type='spatial').
    title : str
        Chart title.
    colors : None or list of str
        Marker colors corresponding to each object.
    labels : list of str
        List of objects labels for the legend.
    invert_xaxis, invert_yaxis : bool
        If True, the axis is inverted.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    xmin, xmax, ymin, ymax : None or scalar
        Limits of the axes.
    output_csv : str or None
        File where to save the data.

    """
    # defining the averaging function
    if score_type == 'temporal':
        sc_function = evt.Evaluator.temporal_scores
        sc_kwargs = dict(availability_ratio=availability_ratio)
    elif score_type == 'spatial':
        sc_function = evt.Evaluator.spatial_scores
        sc_kwargs = dict(min_nb_sta=min_nb_sta)
    else:
        raise evt.EvaltoolsError(
            "score_type argument must be either equal to "
            "'temporal' or 'spatial'."
        )

    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # build 'labels' from objects 'model' attribute if 'labels' is None
    if labels is None:
        labels = [obj.model for obj in objects]

    # define objects colors in case of None
    if colors is None:
        colors = []
        for obj in objects:
            colors.append(obj.color)

    # score computing
    res = {}
    for obj, lab in zip(objects, labels):
        stats = sc_function(
            obj,
            score_list=[xscore, yscore],
            **sc_kwargs,
        )
        stats = stats[f'D{forecast_day}']

        if score_type == 'temporal':
            avail_criteria = (
                np.sum(~stats[xscore].isna(), axis=0) >=
                min_nb_sta
            )
        elif score_type == 'spatial':
            avail_criteria = (
                np.sum(~stats[xscore].isna(), axis=0) >=
                availability_ratio*stats[xscore].shape[0]
            )

        if avail_criteria:
            quartiles = np.nanpercentile(stats, q=[25, 50, 75], axis=0)
            quartiles = pd.DataFrame(
                quartiles,
                index=[25, 50, 75],
                columns=stats.columns,
            )
            res[lab] = quartiles
        else:
            res[lab] = pd.DataFrame(
                np.nan,
                index=[25, 50, 75],
                columns=stats.columns,
            )
            print(
                f"Warning: Availability criteria not fulfilled for {lab} !!!"
            )

    # plotting
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)
    plt.title(title)
    for obj, lab, col in zip(objects, labels, colors):
        rect = plt.Rectangle(
            xy=tuple(res[lab][[xscore, yscore]].loc[25]),
            width=res[lab][xscore].loc[75]-res[lab][xscore].loc[25],
            height=res[lab][yscore].loc[75]-res[lab][yscore].loc[25],
            edgecolor=col,
            fc='none',
            lw=2)
        ax.add_patch(rect)
        plt.gca().add_patch(rect)
        ax.scatter([res[lab][xscore].loc[50]], [res[lab][yscore].loc[50]],
                   s=50, c=col, marker='o', label=lab)

    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_axisbelow(True)
    leg = ax.legend(loc='best', scatterpoints=1, prop={'size': 10})
    leg.draw_frame(False)
    ax.set_xlabel(xscore, fontsize=12)
    ax.set_ylabel(yscore, fontsize=12)
    if invert_xaxis is True:
        plt.gca().invert_xaxis()
    if invert_yaxis is True:
        plt.gca().invert_yaxis()

    # paint in black y=0 and x=0
    if black_axes is True:
        _mpl.draw_black_axes(ax)

    # save data
    if output_csv is not None:
        with open(output_csv, 'w', encoding="utf-8") as f:
            for lab in labels:
                f.write(lab+'\n')
                f.write(res[lab].to_string())
                f.write('\n\n')

    return fig, ax


@plot_func
def plot_time_series(
        objects, station_list=None, start_end=None, forecast_day=0,
        plot_type=None, envelope=False, min_nb_sta=1,
        colors=None, linestyles=None, markers=None, labels=None,
        obs_style=None, title="", ylabel='concentration', xticking='auto',
        date_format='%Y-%m-%d', ymin=None, ymax=None,
        black_axes=False, nb_of_minor_ticks=(2, 2), thresh=None, thresh_kw={},
        output_csv=None, fig=None, ax=None):
    """
    Plot time series of obs and sim at a chosen forecast day.

    By default, there is one line per station and per object.
    Observations are taken from the fisrt object of the list (if of type
    Evaluator).

    Parameters
    ----------
    objects : list of evaltools.Evaluator or evaltools.Simulations objects
        Evaluator or Simulations objects used for plotting.
    station_list : None or list of str
        List of stations to display. If None, all stations of the
        first element of <objects> argument are processed.
    start_end : None or list of two datetime.date objects
        Boundary dates for abscissa.
    forecast_day : int
        Forecast day for which to plot the data.
    plot_type : str
        If 'median' the median of all stations values for a given time
        is plotted. If 'mean' the mean of all stations values for a given
        time is plotted. For any other value, all station values are plotted
        separately.
    envelope : bool
        Only use if plot_type is 'median' or 'mean'. If True, draw quartiles
        one and three around the median curve (case plot_type == 'median')
        or draw +/- the standard deviation around the mean curve (case
        plot_type == 'mean').
    min_nb_sta : int
        Minimal number of value required to compute the median or mean of
        all stations. Ingored if plot_type == None.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    labels : None or list of str
        List of labels for the legend.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    title : str
        Title for the plot.
    ylabel : str
        Label for the y axis.
    xticking : str
        Defines the method used to set x ticks. It can be 'auto' (automatic),
        'mondays' (a tick every monday) or 'daily' (a tick everyday).
    date_format : str
        String format for dates as understood by python module datetime.
    ymin, ymax : None or scalar
        Limits of the axes.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    thresh : None or float
        If not None, a horizontal line y = val is drawn (only if the line
        would appear inside the y axis limits).
    thresh_kw : dict
        Additional keyword arguments passed to matplotlib when drawing the
        threshold line (only used if thresh argument is not None) e.g.
        {'color': '#FFA600'}.
    output_csv : str or None
        File where to save the data. The file name must contain {model}
        instead of the model name.

    """
    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)
    if station_list is None:
        station_list = reduce(
            np.intersect1d,
            [obj.sim_df[0].columns for obj in objects],
        )
    if not obs_style:
        obs_style = {'color': 'k', 'alpha': 0.5}
    else:
        obs_style = deepcopy(obs_style)

    obs_label = obs_style.pop('label', 'observations')

    # first object in the list defines abscissa boundaries
    if start_end is None:
        start_end = [objects[0].start_date, objects[0].end_date]

    # find indices where we have no missing values in every object
    # complex to manage objects with =! step or =! period
    if plot_type in ['median', 'mean']:
        nan_idx = _find_any_missing(
            objects, labels, forecast_day=forecast_day, start_end=start_end,
            station_list=station_list,
        )

    # sim plotting
    handles = []
    for lb, c, ls, m, obj in zip(labels, colors, linestyles, markers, objects):
        sim_df = obj.get_sim(
            forecast_day=forecast_day,
            start_end=start_end,
        )
        handles.append(
            mlines.Line2D(
                [], [],
                color=c,
                label=lb,
                ls='-' if envelope else ls,
            )
        )
        sim_df = sim_df[station_list]
        linestyles = repeat(ls)

        if plot_type in ['median', 'mean']:
            sim_df[nan_idx[lb]] = np.nan

        if plot_type == 'median':
            if envelope is True:
                quant = [.25, .5, .75]
                linestyles = ['--', '-', '--']
            else:
                quant = [.5]
            data = sim_df.quantile(quant, axis=1).T
            idx_not_enough = (~sim_df.isna()).sum(axis=1) < min_nb_sta
            data.loc[idx_not_enough] = np.nan

        elif plot_type == 'mean':
            data = sim_df.mean(axis=1).T
            data.name = 'mean'
            data = data.to_frame()
            idx_not_enough = (~sim_df.isna()).sum(axis=1) < min_nb_sta
            data.loc[idx_not_enough] = np.nan
            if envelope is True:
                linestyles = ['-', '--', '--']
                std = sim_df.std(axis=1).T
                data['up_bnd'] = data['mean'] + std
                data['lo_bnd'] = data['mean'] - std

        else:
            data = sim_df

        fig, ax = _mpl.mpl_plot_line(
            data,
            plot_with_pandas=False,
            plot_kw=dict(
                color=repeat(c),
                linestyle=linestyles,
                marker=repeat(m),
            ),
            legend=False,
            fig=fig, ax=ax,
        )

        if output_csv is not None:
            _write_csv(data, output_csv, model=lb)

    # obs are plotted only for the first object in the list
    if isinstance(objects[0], evt.evaluator.Evaluator):

        obs_df = objects[0].get_obs(
            forecast_day=forecast_day,
            start_end=start_end,
        )
        obs_df = obs_df[station_list]
        handles.append(mlines.Line2D([], [], label=obs_label, **obs_style))
        ls = obs_style.pop('linestyle', obs_style.pop('ls', None))
        obs_style['linestyle'] = repeat(ls)
        m = obs_style.pop('marker', None)
        obs_style['marker'] = repeat(m)
        c = obs_style.pop('color', obs_style.pop('c', None))
        obs_style['color'] = repeat(c)

        if plot_type in ['median', 'mean']:
            obs_df[nan_idx[labels[0]]] = np.nan

        if plot_type == 'median':
            if envelope is True:
                quant = [.25, .5, .75]
                obs_style['linestyle'] = ['--', '-', '--']
            else:
                quant = [.5]
            data = obs_df.quantile(quant, axis=1).T
            idx_not_enough = (~obs_df.isna()).sum(axis=1) < min_nb_sta
            data.loc[idx_not_enough] = np.nan

        elif plot_type == 'mean':
            data = obs_df.mean(axis=1).T
            data.name = 'mean'
            data = data.to_frame()
            idx_not_enough = (~obs_df.isna()).sum(axis=1) < min_nb_sta
            data.loc[idx_not_enough] = np.nan
            if envelope is True:
                obs_style['linestyle'] = ['-', '--', '--']
                std = obs_df.std(axis=1).T
                data['up_bnd'] = data['mean'] + std
                data['lo_bnd'] = data['mean'] - std

        else:
            data = obs_df

        fig, ax = _mpl.mpl_plot_line(
            data,
            plot_with_pandas=False,
            plot_kw=obs_style,
            legend=False,
            fig=fig, ax=ax,
        )

        if output_csv is not None:
            _write_csv(data, output_csv, model=obs_label)

    _mpl.set_axis_elements(
        ax,
        ylabel=ylabel,
        black_axes=black_axes,
        title=title,
        nb_of_minor_ticks=nb_of_minor_ticks,
        xmin=start_end[0],
        xmax=start_end[1],
        ymin=ymin,
        ymax=ymax,
    )

    _mpl.set_time_xaxis(fig, ax, xticking, date_format=date_format)

    # legend
    plt.legend(handles=handles, ncol=legend_ncol)

    # try to control number of yticks
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)

    # draw a hline corresponding to thresh if inside ylim
    if thresh is not None:
        _mpl.hline(ax, thresh, **thresh_kw)

    return fig, ax


@plot_func
def plot_bar_scores(
        objects, score, forecast_day=0, averaging='mean', title="",
        labels=None, colors=None, subregions=None, xtick_labels=None,
        availability_ratio=.75, bar_kwargs={}, ref_line=None,
        output_csv=None, fig=None, ax=None):
    """
    Draw a barplot of scores.

    Draw a barplot with one bar per object. If there are subregions, one set
    of bars per region will be drawn.

    The score are first computed for each measurement sites and then averaged.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Bar colors corresponding to each object.
    subregions : None or list of list of str
        One set of bars per sub-region will be drawn. The sub-regions must be
        given like [['FR'], ['ES', 'FR'], 'all', ...], where 'FR', 'ES', ...
        are the first letters of the station codes you want to keep
        and 'all' means that all stations are kept.
    xtick_labels : None or list of str
        List of labels for the xticks. These labels corresponds to sub-region
        define *subregions* argument. Labels can contain '{nbStations}' that
        will be replaced by the corresponding number of stations used to
        compute the score (warning: if several objects are displayed, the
        number of stations corresponds to the first one).
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute the mean scores.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    ref_line : dict of args or None
        Plot an horizontal line whose arguments are passed to
        matplotlib.pyplot.hline.
    output_csv : str or None
        File where to save the data.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects, False)
    _check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, None, None)
    if subregions is None:
        subregions = ["all"]
    if xtick_labels is None:
        xtick_labels = np.full(len(subregions), '')

    if len(subregions) != len(xtick_labels):
        raise evt.EvaltoolsError(
            "'subregions' and 'xtick_labels' must have same length."
        )

    # stats computing
    stats = [
        obj.temporal_scores(
            score_list=[score],
            availability_ratio=availability_ratio,
        )[f'D{forecast_day}']
        for obj in objects
    ]

    # defining averaging function
    if averaging == 'mean':
        average = pd.DataFrame.mean
    elif averaging == 'median':
        average = pd.DataFrame.median
    else:
        raise evt.EvaltoolsError(
            "averaging argument must be either equal to 'mean' or 'median'."
        )

    df = pd.DataFrame()
    nb_stations = []
    for i, region in enumerate(subregions):
        if region == 'all':
            sub_df = pd.DataFrame(
                {lab: y[score] for lab, y in zip(labels, stats)}
            )
        else:
            kept_stations = [
                sta
                for sta in objects[0].stations.index
                if any(re.match(cn, sta) for cn in region)
            ]
            if len(kept_stations) == 0:
                print(f"Warning: subregion {region} is empty !!!")
            sub_df = pd.DataFrame(
                {
                    lab: y[score].loc[kept_stations]
                    for lab, y in zip(labels, stats)
                }
            )
        nb_stations.append(np.sum(sub_df[labels[0]].notnull()))
        df[i] = average(sub_df)

    df.columns = [
        lab.format(nbStations=nb_stations[i])
        for i, lab in enumerate(xtick_labels)
    ]

    # to respect variable order
    df = df.loc[labels]

    fig, ax = _mpl.mpl_plot_bar(
        df.T,
        plot_kw=dict(
            color=colors,
            legend=False,
            **bar_kwargs,
        ),
        fig=fig,
        ax=ax,
    )

    # save in CSV format
    if output_csv is not None:
        df.T.to_csv(
            path_or_buf=output_csv.format(score=score), sep=" ",
            na_rep="nan",
        )

    # ref line
    if ref_line is not None:
        ref_line['y'] = 0. if 'y' not in ref_line else ref_line['y']
        ref_line['xmin'] = (
            -1. if 'xmin' not in ref_line else ref_line['xmin']
        )
        ref_line['xmax'] = (
            len(df.columns) if 'xmax' not in ref_line else ref_line['xmax']
        )
        ax.hlines(**ref_line)

    # legend
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=c, label=l, linewidth=3)
            for c, l in zip(colors, labels)
        ],
        ncol=legend_ncol,
    )

    _mpl.set_axis_elements(
        ax,
        xticks_kw=dict(rotation=45, ha='right'),
        ylabel=score,
        title=title,
    )

    return fig, ax


@plot_func
def plot_bar_exceedances(
        obj, threshold, data="obs", start_end=None, forecast_day=0,
        labels=None, title="", ylabel=None, ymin=None, ymax=None,
        subregions=None, xticking='daily', date_format='%Y-%m-%d',
        bar_kwargs={}, output_csv=None, fig=None, ax=None):
    """
    Draw a Barplot of threshold exceedances.

    Draw a barplot of threshold exceedances for the period defined by
    start_end. If there are subregions, their bars will be respectively
    drawn one above the other.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Evaluator object used for plotting.
    threshold : scalar
        Threshold value.
    data : str
        Data to be used to compute exceedances. Can be "obs" or "sim".
    start_end : None or list of two datetime.date objects
        Boundary dates for abscissa.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    labels : None or list of str
        List of labels for the legend.
    title : str
        Title for the plot. May contain {total} to print total number of
        obs/sim that exceed threshold.
    ylabel : str
        Ordinate axis label.
    subregions : None or list of list of str
        One set of bars per sub-region will be drawn. The sub-regions must be
        given like [['FR'], ['ES', 'FR'], 'all', ...], where 'FR', 'ES', ...
        are the first letters of the station codes you want to keep
        and 'all' means that all stations are kept.
    xticking : str
        Defines the method used to set x ticks. It can be either 'daily',
        'mondays' or 'bimonthly'.
    date_format : str
        String format for dates as understood by python module datetime.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    ymin, ymax : None or scalar
        Limits of the y axis.
    output_csv : str or None
        File where to save the data.

    """
    # get some useful variables
    if start_end is None:
        start_end = [obj.start_date, obj.end_date]
    if ylabel is None:
        ylabel = tr.bar_exceedances['ylabel'][LANG]

    # subsetting: one object per region
    if subregions is None:
        objects = [obj]
    else:
        objects = [
            obj
            if region == 'all' else obj.select_countries(region, False)
            for region in subregions
        ]

    days_list = evt.dataset.time_range(
        start_end[0],
        start_end[1],
        series_type='daily',
    )
    days_list = days_list.date

    bars_heights = pd.DataFrame()
    for i, objct in enumerate(objects):
        for day in days_list:
            obs = objct.get_obs(forecast_day, (day, day))
            sim = objct.get_sim(forecast_day, (day, day))
            res = evt.scores.contingency_table(
                obs.values.flatten(),
                sim.values.flatten(),
                thr=threshold,
            )
            day_shifted = day + dt.timedelta(days=forecast_day)
            if data == "sim":
                bars_heights.loc[day_shifted, i] = res[2, 0]
            else:
                bars_heights.loc[day_shifted, i] = res[0, 2]

    if labels is not None:
        bars_heights.columns = labels

    fig, ax = _mpl.mpl_plot_bar(
        bars_heights,
        plot_kw=dict(
            stacked=True,
            legend=(labels is not None),
            **bar_kwargs,
        ),
        fig=fig,
        ax=ax,
    )

    # Customisation
    tot = evt.scores.contingency_table(
        obj.get_obs(forecast_day).values.flatten(),
        obj.get_sim(forecast_day).values.flatten(),
        thr=threshold,
    )
    total = tot[2, 0] if data == "sim" else tot[0, 2]

    # set xticks
    if xticking == 'daily':
        xtick_labels = [
            d.strftime(date_format)
            for d in bars_heights.index
        ]
        xtick_pos = list(range(len(days_list)))
    elif xticking == 'mondays':
        xtick_labels = [
            d.strftime(date_format)
            for d in bars_heights.index
            if d.weekday() == 0
        ]
        xtick_pos = [
            i
            for i, d in enumerate(bars_heights.index)
            if d.weekday() == 0
        ]
    elif xticking == 'bimonthly':
        xtick_labels = [
            d.strftime(date_format)
            for d in bars_heights.index
            if d.day in [0, 15]
        ]
        xtick_pos = [
            i
            for i, d in enumerate(bars_heights.index)
            if d.day in [0, 15]
        ]
    else:
        raise evt.EvaltoolsError(
            "xticking argument must be either 'daily', 'mondays' or "
            "'bimonthly'"
        )
    plt.xticks(xtick_pos, xtick_labels)

    _mpl.set_axis_elements(
        ax,
        xticks_kw=dict(rotation=45, ha='right'),
        ylabel=ylabel,
        title=title.format(total=int(total)),
        ymin=ymin,
        ymax=ymax,
    )

    # save in CSV format
    if output_csv is not None:
        bars_heights.to_csv(path_or_buf=output_csv, sep=" ", na_rep="nan")

    return fig, ax


@plot_func
def plot_line_exceedances(
        objects, threshold, start_end=None, forecast_day=0,
        labels=None, colors=None, linestyles=None, markers=None,
        title="", ylabel=None, xticking='daily',
        date_format=None, ymin=None, ymax=None,
        obs_style=None, output_csv=None,
        black_axes=False, nb_of_minor_ticks=(1, 2),
        fig=None, ax=None):
    """
    Plot threshold exceedances over time.

    This function is based on contingency table.
    On each plot, there will be as many lines as there are objects the
    object list, plus one line with the observations of first object.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    threshold : scalar
        Threshold value.
    start_end : None or list of two datetime.date objects
        Boundary dates for abscissa.
    forecast_day : int.
        Integer corresponding to the chosen forecast day used for plotting.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    title : str
        Title for the plot.
    ylabel : str
        Ordinate axis label.
    xticking : str
        Defines the method used to set x ticks. It can be either 'daily',
        'mondays', 'bimonthly' or 'auto'.
    date_format : str
        String format for dates as understood by python module datetime.
    ymin, ymax : None or scalar
        Limits of the y axis.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    output_csv : str or None
        File where to save the data.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.

    """
    # check if objects are defined on the station list
    _check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)
    if ylabel is None:
        ylabel = tr.line_exceedances['ylabel'][LANG]

    if not obs_style:
        obs_style = {'color': 'k', 'alpha': 0.5}
    else:
        obs_style = deepcopy(obs_style)

    obs_label = obs_style.pop('label', 'observations')

    # get period
    if start_end is None:
        _check_period(objects)
        start_end = [objects[0].start_date, objects[0].end_date]
    days_list = evt.dataset.time_range(
        start_end[0],
        start_end[1],
        series_type='daily',
    )
    days_list = days_list.date

    # compute exceedances
    df = pd.DataFrame(
        columns=[obj.model for obj in objects] + [obs_label]
    )
    for obj, lab in zip(objects, labels):
        for day in days_list:
            obs = obj.get_obs(forecast_day, (day, day))
            sim = obj.get_sim(forecast_day, (day, day))
            res = evt.scores.contingency_table(
                obs.values.flatten(),
                sim.values.flatten(),
                thr=threshold,
            )
            day_shifted = day + dt.timedelta(days=forecast_day)
            df.loc[day_shifted, lab] = res[2, 0]
            if obj is objects[0]:
                df.loc[day_shifted, obs_label] = res[0, 2]

        if df[lab].isna().all():
            print(f"*** all nan values for {lab}")

    fig, ax = _mpl.mpl_plot_line(
        df[labels],
        plot_with_pandas=False,
        plot_kw=dict(
            color=colors,
            linestyle=linestyles,
            marker=markers,
        ),
        legend=False,
        fig=fig, ax=ax,
    )

    obs_ls = obs_style.pop('linestyle', obs_style.pop('ls', None))
    obs_style['linestyle'] = [obs_ls]
    obs_m = obs_style.pop('marker', None)
    obs_style['marker'] = [obs_m]
    obs_c = obs_style.pop('color', obs_style.pop('c', None))
    obs_style['color'] = [obs_c]
    fig, ax = _mpl.mpl_plot_line(
        df[[obs_label]],
        plot_with_pandas=False,
        plot_kw=obs_style,
        legend_type='ax',
        fig=fig, ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        ylabel=ylabel,
        black_axes=black_axes,
        title=title,
        # xmin=start_end[0],
        # xmax=start_end[1],
        ymin=ymin,
        ymax=ymax,
        nb_of_minor_ticks=nb_of_minor_ticks,
    )

    _mpl.set_time_xaxis(fig, ax, xticking, date_format=date_format)

    # save data
    if output_csv is not None:
        df.to_csv(path_or_buf=output_csv, sep=" ", na_rep="nan")

    return fig, ax


@plot_func
def plot_bar_contingency_table(
        objects, threshold, forecast_day=0, start_end=None, title="",
        labels=None, ymin=None, ymax=None, bar_kwargs=None,
        output_csv=None, fig=None, ax=None):
    """
    Draw a barplot from the contingency table.

    For each object, draw a bar for good detections, false
    alarms, and missed alarms.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    threshold : scalar
        Threshold value.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    start_end : None or list of two datetime.date objects
        Boundary dates for the studied period.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels coresponding to each object.
    ymin, ymax : None or scalar
        Limits of the y axis.
    output_csv : str or None
        File where to save the data.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # define objects labels in case of None
    if labels is None:
        labels = []
        for obj in objects:
            labels.append(obj.model)

    # stats computing
    df = pd.DataFrame(index=tr.bar_contingency_table['index'][LANG])
    for obj, lab in zip(objects, labels):
        obs = obj.get_obs(forecast_day=forecast_day, start_end=start_end)
        sim = obj.get_sim(forecast_day=forecast_day, start_end=start_end)
        tab = evt.scores.contingency_table(
            obs.values.flatten(),
            sim.values.flatten(),
            thr=threshold,
        )
        df[lab] = [tab[0, 0], tab[1, 0], tab[0, 1]]

    # barplot
    plot_kw = dict(color=['#2ca02c', '#ff7f0e', '#d62728'])
    bar_kwargs = bar_kwargs or {}
    plot_kw.update(bar_kwargs)
    fig, ax = _mpl.mpl_plot_bar(
        df.T,
        plot_kw=plot_kw,
        fig=fig,
        ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title=title,
        ymin=ymin,
        ymax=ymax,
    )

    # save in CSV format
    if output_csv is not None:
        df.T.to_csv(path_or_buf=output_csv, sep=" ", na_rep="nan")

    return fig, ax


@plot_func
def plot_bar_scores_conc(
        objects, score, conc_range, forecast_day=0, averaging='mean',
        title=None, labels=None, colors=None, xtick_labels=None,
        min_nb_val=10, based_on='obs', bar_kwargs={},
        nb_vals=True, output_csv=None, fig=None, ax=None):
    """
    Barplot for scores per concentration class.

    Data is grouped depending on the desired concentration classes, then
    scores are computed for each site and averaged.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    conc_range : list of scalars
        List used to determine concentration intervals on which to compute
        scores. Must contain at least two values. e.g [25, 45, 80] determines
        scores with concentrations between 25 and 45, and between 45 and 80.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Bar colors corresponding to each object.
    xtick_labels : None or list of str
        List of labels for the xticks.
    min_nb_val : int
        Minimal number of (obs, sim) couple required for a score to be
        computed.
    based_on : str
        If 'sim', concentrations are determined from simulation data. Else
        ('obs') they are determined with observations.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    nb_vals : boolean
        Whether the number of computed values for each bar must be displayed
        or not.
    output_csv : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects, False)
    _check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, None, None)

    # stats computing
    stats = []
    tot = []
    for i in range(len(conc_range)-1):
        stats.append([])
        tot.append([])
        stats[i], tot[i] = zip(
            *[
                obj.conc_scores(
                    score_list=[score],
                    conc_range=[conc_range[i], conc_range[i+1]],
                    min_nb_val=min_nb_val,
                    based_on=based_on,
                    forecast_day=forecast_day,
                )
                for obj in objects
            ]
        )

    if nb_vals is True:
        total = [tot[j][0] for j in range(len(tot))]
        total = np.round(np.true_divide(total, sum(total))*100, decimals=1)
        total = [str(t)+'%' for t in total]
        tot = str(sum(tot[:][0]))

    if xtick_labels is None:
        xtick_labels = [
            str(c1) + ' <= c < ' + str(c2)
            for c1, c2 in zip(conc_range[:-1], conc_range[1:])
        ]
        # xtick_labels[-1] = str(conc_range[-2])+' <= c'

    # defining averaging function
    if averaging == 'mean':
        average = pd.DataFrame.mean
    elif averaging == 'median':
        average = pd.DataFrame.median
    else:
        raise evt.EvaltoolsError(
            "averaging argument must be either equal to 'mean' or 'median'."
        )

    df = pd.DataFrame()
    for i in range(len(conc_range)-1):
        sub_df = pd.DataFrame({lab: y[score]
                               for lab, y in zip(labels, stats[i])})
        df[i] = average(sub_df)

    # set tick labels
    df.columns = [xtick_labels[i] for i in range(len(conc_range)-1)]

    # to respect objects order
    df = df.loc[labels]

    # barplot
    fig, ax = _mpl.mpl_plot_bar(
        df.T,
        plot_kw=dict(
            color=colors,
            legend=False,
            **bar_kwargs,
        ),
        fig=fig,
        ax=ax,
    )

    _mpl.set_axis_elements(
        ax,
        title_kw=dict(label=title, pad=25),
        ylabel=score,
        xticks_kw=dict(rotation=30, ha='right'),
    )

    if output_csv is not None:
        df.T.to_csv(output_csv.format(score=score), sep=' ', na_rep='nan')

    # print tot values
    if nb_vals is True:
        pos = [1./len(total)/2.]
        for x in range(len(total)-1):
            pos.append(pos[-1]+1./len(total))
        for x in range(len(total)):
            ax.text(
                pos[x], 1, str(total[x]),
                fontsize=8, color="#7A7A7A",
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
            )
        plt.annotate(
            tr.bar_scores_conc['annotation'][LANG].format(tot),
            xy=(0, 1),
            xytext=(0, 17),
            xycoords='axes fraction',
            textcoords='offset points',
            va='top',
            fontsize=8,
            fontstyle='italic',
            color='#7A7A7A',
        )

    # legend
    handles = [mlines.Line2D([], [], color=c, label=l, linewidth=3)
               for c, l in zip(colors, labels)]
    plt.legend(
        handles=handles,
        ncol=legend_ncol,
    )

    return fig, ax


@plot_func
def plot_roc_curve(
        objects, thresholds, forecast_day=0,
        labels=None, colors=None, markers=None,
        title="ROC diagram", xlabel=None, ylabel=None, start_end=None,
        fig=None, ax=None):
    """
    Draw the ROC curve for each object.

    A ROC curve is created by plotting the true positive rate against the
    false positive rate at various threshold values. Thus, it can be useful
    to assess model performance regarding threshold exceedances.
    In the legend, SSr (Skill Score Ratio) corresponds to the Gini
    coefficient which is the area between the ROC curve and the
    no-discrimination line multiplied by two.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    thresholds : list of scalars
        Threshold values.
    forecast_day : int
        Forecast day for which to use data to compute the ROC diagram.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Marker colors corresponding to each object.
    markers : None or list of str
        Marker shapes corresponding to each object.
    title : str
        Chart title.
    xlabel : str
        Label for the x axis.
    ylabel : str
        Label for the y axis.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects, False)
    _check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, None, markers, 'x')
    xlabel = xlabel or 'POFD'
    ylabel = ylabel or 'POD'

    # get performance scores for each object
    pod = {}
    pofd = {}
    for obj, lab in zip(objects, labels):
        prob_detect = [0.]
        prob_false_detect = [0.]
        for thr in reversed(np.sort(thresholds)):
            obs = obj.get_obs(forecast_day, start_end=start_end)
            sim = obj.get_sim(forecast_day, start_end=start_end)
            perf = evt.scores.contingency_table(
                obs.values.flatten(),
                sim.values.flatten(),
                thr=thr,
            )
            prob_detect.append(
                perf[0, 0]/perf[0, 2] if perf[0, 2] != 0 else np.nan
            )
            prob_false_detect.append(
                perf[1, 0]/perf[1, 2] if perf[1, 2] != 0 else np.nan
            )
        prob_detect.append(1.)
        prob_false_detect.append(1.)
        pod[lab] = prob_detect
        pofd[lab] = prob_false_detect

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    # reference x=y line
    ax.plot([0, 1], [0, 1], 'ko-', label='No skill (SSr=0)')

    # plot points and lines for each object
    for i, lab in enumerate(labels):
        area = np.trapz(y=pod[lab], x=pofd[lab]) / 0.5 - 1.0
        area = str(round(area, 2))
        ax.plot(
            pofd[lab], pod[lab],
            marker=markers[i],
            color=colors[i],
            label=labels[i] + f" (SSr={area})",
        )

    ax.legend(loc='lower right')

    _mpl.set_axis_elements(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    return fig, ax


@plot_func
def plot_performance_diagram(
        objects, threshold, forecast_day=0,
        labels=None, colors=None, markers=None,
        title="Performance Diagram",
        start_end=None, fig=None, ax=None):
    """
    Plot a performance diagram.

    This diagram is created by plotting the probability of detection (true
    positive rate) against the success ratio (positive predictive value)
    relative to the chosen detection threshold. It also displays the critical
    success index (threat score) and the frequency bias.

    References
    ----------
        Roebber, P.J., 2009: Visualizing multiple measures of forecast quality.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    threshold : scalar
        Threshold value.
    forecast_day : int
        Forecast day for which to use data to compute the diagram.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Marker colors corresponding to each object.
    markers : None or list of str
        Marker shapes corresponding to each object.
    title : str
        Diagram title.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects, False)
    _check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, None, markers, 'o')
    xlabel = tr.performance_diagram['xlabel'][LANG]
    ylabel = tr.performance_diagram['ylabel'][LANG]
    csi_label = tr.performance_diagram['csi_label'][LANG]
    freq_label = tr.performance_diagram['freq_label'][LANG]

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    ax.grid(False, which='both')

    # get performance scores for each object
    prob_detect = []
    success_ratio = []
    for obj in objects:
        perf = evt.scores.contingency_table(
            obj.get_obs(forecast_day, start_end=start_end).values.flatten(),
            obj.get_sim(forecast_day, start_end=start_end).values.flatten(),
            thr=threshold,
        )
        prob_detect.append(
            perf[0, 0]/perf[0, 2] if perf[0, 2] != 0 else np.nan
        )
        success_ratio.append(
            perf[0, 0]/perf[2, 0] if perf[2, 0] != 0 else np.nan
        )

    # grid for background colors and lines
    grid_ticks = np.arange(0, 1.001, 0.001)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)

    # critical success index background
    csi = np.zeros_like(sr_g)
    mask = (sr_g*pod_g == 0)
    csi[~mask] = 1.0/(1.0/sr_g[~mask] + 1.0/pod_g[~mask] - 1.0)

    # define custom cmap to apply alpha to avoid bug "Colorbar extend patches
    # do not have correct alpha"
    cmap = plt.cm.RdYlGn
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = .5
    my_cmap = ListedColormap(my_cmap)
    csi_contour = plt.contourf(
        sr_g, pod_g, csi, np.arange(0., 1.01, 0.05),
        # extend="max",
        cmap=my_cmap,
        # alpha=0.5,
    )
    cbar = plt.colorbar(csi_contour)

    # frequency bias lines
    bias = np.full_like(sr_g, np.nan)
    mask = (sr_g == 0)
    bias[~mask] = pod_g[~mask]/sr_g[~mask]
    freq_lines = plt.contour(
        sr_g, pod_g, bias,
        [0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2, 4],
        colors="k", linestyles="dashed",
    )
    plt.clabel(
        freq_lines,
        fmt="%1.1f",
        manual=[
            (.8, .09), (.85, .21), (.9, .45), (.92, .7), (.94, .94),
            (.75, .95), (.6, .92), (.44, .89), (.21, .83),
        ],
    )
    handles = [
        mlines.Line2D(
            [], [],
            color="k",
            label=freq_label,
            # linewidth=3,
            ls="dashed",
        )
    ]

    # plot each point
    for i, obj in enumerate(objects):
        ax.scatter(
            success_ratio[i], prob_detect[i],
            marker=markers[i],
            c=colors[i],
            label=labels[i],
            zorder=10,
        )
        handles.append(
            mlines.Line2D(
                [], [],
                color=colors[i],
                label=labels[i],
                marker=markers[i],
                ls='',
            )
        )

    _mpl.set_axis_elements(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    cbar.set_label(csi_label, fontsize=13)

    legend_params = dict(
        bbox_to_anchor=(1.3, 1), loc='upper left',
        # fontsize=12, framealpha=1, frameon=True,
        handles=handles,
    )
    plt.legend(**legend_params)

    return fig, ax


@plot_func
def plot_diurnal_cycle(
        objects, station_list=None, availability_ratio=0.75,
        colors=None, linestyles=None, markers=None,
        labels=None, title="", xlabel=None,
        ylabel='concentration', ymin=None, ymax=None,
        normalize=False, plot_type=None, envelope=False,
        black_axes=False, obs_style=None,
        nb_of_minor_ticks=(3, 1), output_csv=None, fig=None, ax=None):
    """
    Plot the diurnal cycle of observations and simulations.

    On each plot, there is one line per station and per object.
    Observations are taken from the first object of the list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    station_list : None or list of str
        List of stations to display. If None, all stations
        of the first element of <objects> argument are
        processed.
    colors : None or list of str
        Line colors corresponding to each object.
    linestyles : None or list of str
        Line styles corresponding to each object.
    markers : None or list of str
        Line markers corresponding to each object.
    labels : None or list of str
        List of labels for the legend.
    title : str
        Title for the plot.
    xlabel, ylabel : str
        Label for x and y axes.
    ymin, ymax : None or scalar
        Limits of the axes.
    normalize : bool
        If True, values are normalized for each station by substracting the
        mean and dividing by the standard deviation of the diurnal cycle
        values for the station.
    plot_type : str
        If 'median' the median of all stations values at a forecast hour
        is plotted. If 'mean' the mean of all stations values at a forecast
        hour is plotted. For any other value, all station values are plotted
        separately.
    envelope : bool
        Only use if plot_type is 'median' or 'mean'. If True, draw quartiles
        one and three around the median curve (case plot_type == 'median')
        or draw +/- the standard deviation around the mean curve (case
        plot_type == 'mean').
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    output_csv : str or None
        File where to save the data. The file name must contain {model}
        instead of the model name.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects)
    _check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, linestyles, markers)
    if station_list is None:
        station_list = reduce(np.intersect1d,
                              [obj.stations.index for obj in objects])
    if xlabel is None:
        xlabel = tr.diurnal_cycle['xlabel'][LANG]

    if not obs_style:
        obs_style = {'color': 'k', 'alpha': 0.5}
    else:
        obs_style = deepcopy(obs_style)

    obs_label = obs_style.pop('label', 'observations')

    # sim plotting
    handles = []
    for lb, c, ls, m, obj in zip(labels, colors, linestyles, markers, objects):
        df = obj.temporal_ft_scores(
            score_list=['sim_median'],
            availability_ratio=availability_ratio,
        )
        df = df['sim_median'].loc[station_list]
        handles.append(mlines.Line2D([], [], color=c, label=lb))
        linestyles = repeat(ls)

        if normalize is True:
            df = df.subtract(df.mean(axis=1), axis=0)
            df = df.divide(df.std(axis=1), axis=0)

        if plot_type == 'median':
            if envelope is True:
                quant = [.25, .5, .75]
                linestyles = ['--', '-', '--']
            else:
                quant = [.5]
            data = df.quantile(quant, axis=0).T

        elif plot_type == 'mean':
            data = df.mean(axis=0)
            if envelope is True:
                linestyles = ['-', '--', '--']
                df_std = df.std(axis=0)
                data['up_bnd'] = data['mean'] + df_std
                data['lo_bnd'] = data['mean'] - df_std

        else:
            data = df.T

        fig, ax = _mpl.mpl_plot_line(
            data,
            plot_with_pandas=False,
            plot_kw=dict(
                color=repeat(c),
                linestyle=linestyles,
                marker=repeat(m),
            ),
            legend=False,
            fig=fig, ax=ax,
        )

        if output_csv is not None:
            _write_csv(data, output_csv, model=lb)

    # obs are plotted only for the first object in the list
    obj = objects[0]
    df = obj.temporal_ft_scores(
        score_list=['obs_median'],
        availability_ratio=availability_ratio,
    )
    df = df['obs_median'].loc[station_list]
    handles.append(mlines.Line2D([], [], label=obs_label, **obs_style))
    ls = obs_style.pop('linestyle', obs_style.pop('ls', None))
    obs_style['linestyle'] = repeat(ls)
    m = obs_style.pop('marker', None)
    obs_style['marker'] = repeat(m)
    c = obs_style.pop('color', obs_style.pop('c', None))
    obs_style['color'] = repeat(c)

    if normalize is True:
        df = df.subtract(df.mean(axis=1), axis=0)
        df = df.divide(df.std(axis=1), axis=0)

    if plot_type == 'median':
        if envelope is True:
            quant = [.25, .5, .75]
            obs_style['linestyle'] = ['--', '-', '--']
        else:
            quant = [.5]
        data = df.quantile(quant, axis=0).T

    elif plot_type == 'mean':
        data = df.mean(axis=0).to_frame()
        data.name = 'mean'
        data = data.to_frame()
        if envelope is True:
            linestyles = ['-', '--', '--']
            df_std = df.std(axis=0)
            data['up_bnd'] = data['mean'] + df_std
            data['lo_bnd'] = data['mean'] - df_std

    else:
        data = df.T

    fig, ax = _mpl.mpl_plot_line(
        data,
        plot_with_pandas=False,
        plot_kw=obs_style,
        legend=False,
        fig=fig, ax=ax,
    )

    if output_csv is not None:
        df.to_csv(
            output_csv.format(model=obs_label), sep=' ',
            na_rep='nan', float_format='%g', header=True, index=True,
            date_format='%Y%m%d%H',
        )

    x_max = obj.forecast_horizon*24
    _mpl.set_axis_elements(
        ax,
        xticks_kw=dict(ticks=np.arange(0, x_max, 12)),
        xlabel=xlabel,
        ylabel=ylabel,
        black_axes=black_axes,
        title=title,
        nb_of_minor_ticks=nb_of_minor_ticks,
        ymin=ymin,
        ymax=ymax,
        xmin=0,
        xmax=x_max-1,
    )

    # legend
    plt.legend(handles=handles, ncol=legend_ncol)

    # try to control number of yticks
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)

    return fig, ax


@plot_func
def plot_comparison_scatter_plot(
        score, xobject, yobject, forecast_day=0, title="", xlabel=None,
        ylabel=None, availability_ratio=0.75, nb_outliers=5,
        black_axes=False, color_by=None,
        output_csv=None, fig=None, ax=None):
    """
    Scatter plot to compare two Evaluator objects.

    One score is calculated for two Evaluator objects and used as xy
    coordinates. Points are colored according to the density of points.

    Parameters
    ----------
    score : str
        Computed score.
    xobject, yobject : evaltools.Evaluator object
        Objects used for plotting.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    title : str
        Title for the plots.
    xlabel, ylabel : str
        Labels corresponding to xobject and yobject used for axis labels.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute station scores.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    color_by : None or dictionary
        Dictionary with keys corresponding to station names and
        values corresponding to colors.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    output_csv : str or None
        File where to save the data.

    """
    # check if objects are defined on the same period and same stations
    _check_period([xobject, yobject])
    _check_station_list([xobject, yobject])

    # build 'labels' from objects 'model' attribute if 'labels' is None
    if xlabel is None:
        xlabel = xobject.model
    if ylabel is None:
        ylabel = yobject.model

    # score computing
    stations = np.intersect1d(xobject.stations.index, yobject.stations.index)
    stats = pd.DataFrame(index=stations, columns=[xlabel, ylabel])
    for obj, lab in zip([xobject, yobject], [xlabel, ylabel]):
        df = obj.temporal_scores(
            score_list=[score],
            availability_ratio=availability_ratio,
        )
        df = df[f'D{forecast_day}'][score]
        df.name = lab
        df = pd.DataFrame(df)
        stats.update(df)

    # min max values
    minval = np.nanmin(stats.values.astype(float))
    maxval = np.nanmax(stats.values.astype(float))

    # distance to y = x
    stats['d'] = abs(stats[xlabel]-stats[ylabel])/np.sqrt(2)

    # define point color in stats['z']
    if color_by is None:
        # remove nan to compute gaussian_kde
        idx_not_nan = ~stats.isna().any(axis=1)
        xy = stats.loc[idx_not_nan].values.T.astype('float')
        try:
            z = gaussian_kde(xy)(xy)
            z = pd.DataFrame(z, index=idx_not_nan[idx_not_nan].index,
                             columns=['z'])
            stats['z'] = np.nan
            stats.update(z)
            # sort points by density (=> densest points are plotted last)
            stats.sort_values(by='z', inplace=True)
        except (np.linalg.linalg.LinAlgError, ValueError):
            stats['z'] = 'blue'
    else:
        stats['z'] = pd.Series(color_by)
        stats.sort_values(by='z', inplace=True)

    # plotting
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    ax.scatter(
        stats[xlabel], stats[ylabel],
        c=stats['z'], s=50, edgecolor=None, zorder=10,
    )

    # outliers highlighting
    stats.sort_values(by='d', ascending=False, inplace=True)
    outliers = stats[:nb_outliers]
    ax.scatter(
        outliers[xlabel], outliers[ylabel],
        s=100, facecolors='none', edgecolors='0.5',
    )

    # line y = x
    xf = np.linspace(minval, maxval, 1000)
    plt.plot(xf, xf, 'k--', zorder=11)

    # layout
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('equal')
    ax.set_axisbelow(True)
    if minval < 0 < maxval:
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')

    # paint in black y=0 and x=0
    if black_axes is True:
        fig.canvas.draw()
        _mpl.draw_black_axes(ax)

    # save data
    if output_csv is not None:
        stats.__delitem__('z')
        stats = stats.merge(xobject.stations, how='left', left_index=True,
                            right_index=True, sort=False)
        stats.to_csv(
            output_csv, sep=',', na_rep='nan',
            float_format='%g', header=True, index=True,
            date_format='%Y%m%d%H',
        )

    return fig, ax


@plot_func
def plot_significant_differences(
        score_list, former_objects, later_objects,
        score_type='temporal', forecast_day=0, title="",
        xlabels=None, ylabels=None, availability_ratio=0.75, min_nb_sta=10,
        fig=None, ax=None):
    """
    Chart showing the significativity of differences between simulations.

    Statictical tests are applied to temporal or spatial score values.
    Differences between the two simulations are considered significant if
    both tests are significant (ttest_ind : H0 = the two samples have the
    same mean; mannwhitneyu : H0 = the two samples have the same distribution).
    If one of the tests is not significant, the cell is yellow (value = 0).
    Else, the value is computed from the 9 percentiles (10, ..., 90) of
    each samples :

        value = sum_i(sign(p_i - q_i)*1) for PearsonR,  SpearmanR or FactOf2

        value = sum_i(sign(abs(p_i) - abs(q_i))*1) for other scores

    with (p_i) et (q_i) the percentiles of the two samples.

    Parameters
    ----------
    score_list : list of str
        List of scores for which investigating differences.
    former_objects, later_objects : list of evaltools.Evaluator objects
        Two list of objects that are compared.
    score_type : str
        Score computing method selected from 'temporal' or 'spatial'.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    xlabels, ylabels : None or list of str
        Labels for the x axis (equal to score_list if None) and labels
        for the y axis (corresponding to each object couple comparison
        and equal to the model name of the first object of each couple
        if None).
    availability_ratio : float
        Minimum required rate of available data over the period to
        calculate the scores (only used if score_type is 'temporal').
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute the scores (only used if
        score_type is 'spatial').
    annotation : str or None
        Additional information to write in figure's upper left corner.

    """
    # defining the averaging function
    if score_type == 'temporal':
        sc_function = evt.Evaluator.temporal_scores
        sc_kwargs = dict(availability_ratio=availability_ratio)
    elif score_type == 'spatial':
        sc_function = evt.Evaluator.spatial_scores
        sc_kwargs = dict(min_nb_sta=min_nb_sta)
    else:
        raise evt.EvaltoolsError(
            "score_type argument must be either equal to "
            "'temporal' or 'spatial'."
        )

    # build 'labels' from objects 'model' attribute if 'labels' is None
    if xlabels is None:
        xlabels = score_list
    if ylabels is None:
        ylabels = [obj.model for obj in former_objects]

    percentlist = range(10, 100, 10)

    # tests performing
    difftests = {}
    for xobject, yobject, lab in zip(former_objects, later_objects, ylabels):
        difftests[lab] = {}
        # score computing
        stats_x = sc_function(
            xobject,
            score_list=score_list,
            **sc_kwargs,
        )
        stats_y = sc_function(
            yobject,
            score_list=score_list,
            **sc_kwargs,
        )
        stats_x = stats_x[f'D{forecast_day}']
        stats_y = stats_y[f'D{forecast_day}']
        if score_type == 'temporal':
            stations = np.intersect1d(
                xobject.stations.index,
                yobject.stations.index,
            )
            stats_x = stats_x.loc[stations]
            stats_y = stats_y.loc[stations]

        for sc in score_list:
            idx_not_nan = ~np.logical_or(stats_x[sc].isna(),
                                         stats_y[sc].isna())
            xval = stats_x[sc].loc[idx_not_nan]
            yval = stats_y[sc].loc[idx_not_nan]
            px = [scoreatpercentile(xval.loc[idx_not_nan], q)
                  for q in percentlist]
            py = [scoreatpercentile(yval, q)
                  for q in percentlist]
            chg = 0

            score_obj = evt.scores.get_score(sc)
            if score_obj.sort_key == evt.scores._sort_keys.identity \
                    or score_obj.sort_key == evt.scores._sort_keys.absolute:
                for x, y in zip(px, py):
                    if abs(y) < abs(x):
                        chg += 1
                    elif abs(y) > abs(x):
                        chg += -1
            elif score_obj.sort_key == evt.scores._sort_keys.negate:
                for x, y in zip(px, py):
                    if y > x:
                        chg += 1
                    elif y < x:
                        chg += -1
            else:
                raise evt.EvaltoolsError(
                    f"{sc} is not allowed in plot_significant_differences."
                )

            # Welch T-test on means (independant samples with != variance)
            # Mann-Whitney U-test (no gaussian hypotethis)
            _, p1 = ttest_ind(xval, yval, equal_var=False)
            _, p2 = mannwhitneyu(xval, yval)
            tpl = (len(xval), chg, p1, p2)
            difftests[lab][sc] = tpl

    # choosing colors according to the tests results
    xlab = []
    lx = len(score_list)
    ly = len(ylabels)
    matx = np.zeros((ly, lx))
    for s in range(ly):
        for i in range(lx):
            slab = ylabels[s]
            ilab = score_list[i]
            if (difftests[slab][ilab][2] <= 0.05 and
                    difftests[slab][ilab][3] <= 0.05):
                matx[s, i] = difftests[slab][ilab][1]
        xlab.append(slab + ' (#' + str(difftests[slab][ilab][0]) + ')')

    # plotting
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    plt.pcolor(matx, cmap='RdYlGn', vmin=-(len(percentlist) + 1),
               vmax=len(percentlist) + 1)
    plt.colorbar()
    tick_x = np.arange(0.5, lx + 0.5)
    tick_y = np.arange(0.5, ly + 0.5)
    plt.xticks(tick_x, score_list, rotation=45, ha='right')
    plt.yticks(tick_y, xlab)
    plt.title(title)

    return fig, ax


@plot_func
def plot_values_scatter_plot(
        obj, station_list=None, start_end=None, forecast_day=0, title="",
        xlabel='observations', ylabel=None, black_axes=False, color_by=None,
        group_by=None, xmin=None, xmax=None, ymin=None, ymax=None,
        output_csv=None, fig=None, ax=None):
    """
    Scatter plot to compare directly observations and simulations.

    By default, points are colored according to the density of points.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Object used for plotting.
    station_list : None or list of str
        List of stations to display. If None, all stations of the
        first element of <objects> argument are processed.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    xlabel, ylabel : str
        Labels for x and y axes.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    color_by : None or dictionary
        Dictionary with keys corresponding to station names and
        values corresponding to colors.
    group_by : None or str
        If equal to 'time', the median of all stations is displayed for
        each time. If equal to 'station', the median of all times is
        displayed for each station. Otherwise, one point is plotted for
        each time of each station.
    xmin, xmax, ymin, ymax : None or scalar
        Limits of the axes.
    output_csv : str or None
        File where to save the data.

    """
    # build 'ylabel' from the object's 'model' attribute if it is None
    if ylabel is None:
        ylabel = obj.model

    # all stations by default
    if station_list is None:
        station_list = obj.stations.index

    # get data
    obs = obj.get_obs(forecast_day=forecast_day, start_end=start_end)
    obs = obs[station_list]
    sim = obj.get_sim(forecast_day=forecast_day, start_end=start_end)
    sim = sim[station_list]

    data = pd.DataFrame({xlabel: obs.unstack(), ylabel: sim.unstack()})
    data.index.names = ['id', 'time']

    # compute median
    if group_by == 'station':
        data = data.groupby('id').median()
    elif group_by == 'time':
        data = data.groupby('time').median()

    # distance to y = x
    data['d'] = abs(data[xlabel]-data[ylabel])/np.sqrt(2)

    # save data
    if output_csv is not None:
        with open(output_csv, 'w', encoding="utf-8") as f:
            f.write(data.to_string())

    # min max values
    minval = np.nanmin(data)
    maxval = np.nanmax(data)

    # define point color in data['z']
    if color_by is None:
        # remove nan to compute gaussian_kde
        idx_not_nan = ~data.isna().any(axis=1)
        xy = data[[xlabel, ylabel]].loc[idx_not_nan].values.T.astype('float')
        try:
            z = gaussian_kde(xy)(xy)
            z = pd.DataFrame(z, index=idx_not_nan[idx_not_nan].index,
                             columns=['z'])
            data['z'] = np.nan
            data.update(z)
            # sort points by density (=> densest points are plotted last)
            data.sort_values(by='z', inplace=True)
        except np.linalg.linalg.LinAlgError:
            data['z'] = 'blue'
    else:
        data['z'] = pd.Series(color_by)
        data.sort_values(by='z', inplace=True)

    # plotting
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    ax.scatter(
        data[xlabel], data[ylabel], c=data['z'], s=50,
        edgecolors='none', zorder=10,
    )

    # line y = x
    xf = np.linspace(minval, maxval, 1000)
    ax.plot(xf, xf, 'k--', zorder=11)

    _mpl.set_axis_elements(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        black_axes=black_axes,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    # layout
    if xmin is None and xmax is None and ymin is None and ymax is None:
        plt.axis('equal')

    ax.set_axisbelow(True)

    return fig, ax


@plt.rc_context({'figure.autolayout': False})
def plot_summary_bar_chart(
        objects_lol, forecast_day=0, averaging='mean', mean_obs=True,
        title="", labels_list=None, colors_list=None, groups_labels=None,
        output_file=None, file_formats=['png'],
        availability_ratio=.75, adapt_size=0.25, ncol=2,
        subplots_adjust={'top': 0.85}, fig=None, ax=None):
    """
    Synthetic plot of scores.

    For each object, plots a bar for RMSE and lollipops for bias and
    correlation. Different groups of objects are separated by white space.

    Scores are first computed for each measurement sites and then averaged.

    Parameters
    ----------
    objects_lol : list of lists of evaltools.Evaluator objects
        Evaluator objects used for plotting. One list per group.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    mean_obs : boolean
        Whether to represent mean obs concentration or not.
    title : str
        Title for the figure.
    labels_list : None or list of str
        List of labels for the legend.
    colors_list : None or list of str
        Bar colors corresponding to each object.
    groups_labels : None or list of str
        List of labels for the groups.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute the mean scores.
    adapt_size : float
        Coefficient to increase or reduce subplots width.
    ncol : int
        Number of columns in legend.
    subplots_adjust : dict
        Keyword arguments passed to matplotlib.pyplot.subplots_adjust.

    """
    def _align_yaxis(ax1, v1, ax2, v2, y2max):
        """
        Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1.

        y2max is the maximum value in secondary ax.

        """
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        scale = 1
        while scale*(maxy+dy) < y2max:
            scale += 0.05
        ax2.set_ylim(scale*(miny+dy), scale*(maxy+dy))

    figsize = plt.rcParams['figure.figsize']
    fig = fig or plt.figure(
        figsize=(figsize[0]*len(objects_lol[0])*adapt_size, figsize[1]),
    )
    main_ax = ax or fig.add_subplot(1, 1, 1)
    main_ax.clear()
    main_ax.axis('off')

    axs = []
    n_sub_axes = len(objects_lol)
    for i in range(n_sub_axes):
        axs.append(
            main_ax.inset_axes(
                [i/n_sub_axes, 0, 1/n_sub_axes, 1]
            )
        )
    for i in range(1, n_sub_axes):
        axs[i].sharey(axs[0])

    # fig, axs = plt.subplots(
    #     1, len(objects_lol),
    #     sharex=False, sharey=True,
    #     figsize=(figsize[0]*len(objects_lol[0])*adapt_size, figsize[1]),
    # )
    # if not isinstance(axs, np.ndarray):  # when len(objects_lol) == 1
    #     axs = [axs]

    axs2 = [sax.twinx() for sax in axs]  # ax for correlation

    colors_legend = []
    labels_legend = []
    i, j = -1, -1

    for ax, ax2, objects in zip(axs, axs2, objects_lol):
        # calculation of necessary scores
        stats = [
            obj.temporal_scores(
                score_list=['RMSE', 'MeanBias', 'PearsonR', 'obs_mean'],
                availability_ratio=availability_ratio,
            )[f'D{forecast_day}']
            for obj in objects
        ]

        # defining averaging function
        if averaging == 'mean':
            average = pd.Series.mean
        elif averaging == 'median':
            average = pd.Series.median
        else:
            raise evt.EvaltoolsError(
                "averaging argument must be either equal " +
                "to 'mean' or 'median'.")

        # Get colors
        # User may set less colors and labels than len(objects_lol)
        if colors_list is None:
            colors = [obj.color for obj in objects]
        else:
            colors = []
            for o in objects:
                i = i+1 if i+1 < len(colors_list) else 0
                colors.append(colors_list[i])

        # Get colors and labels for legend
        j += 1
        for k, c in enumerate(colors):
            if c not in colors_legend:  # Keep only unset colors for legend
                colors_legend.append(c)
                if labels_list is None:
                    labels_legend.append(objects[k].model)
                else:
                    if j*len(colors)+k < len(labels_list):
                        labels_legend.append(labels_list[j*len(colors)+k])
                    else:
                        del colors_legend[-1]  # keep only colors with labels

        # RMSE barplot
        df_rmse = pd.DataFrame(
            data=[[average(stats[o]['RMSE']) for o in range(len(objects))]],
            columns=[o.model for o in objects],
        )
        df_rmse.T.plot.bar(ax=ax, legend=False, width=1)
        for c, p in zip(colors, ax.patches):
            p.set_facecolor(c)

        # Bias lollipop plot
        df_bias = pd.DataFrame(
            data=[[average(stats[o]['MeanBias'])
                  for o in range(len(objects))]],
            columns=[o.model for o in objects],
        )
        # loc = np.linspace(-0.5*(1-1./len(objects)),
        #                   0.5*(1-1./len(objects)),
        #                   num=len(objects))
        for i, model in enumerate(df_bias.columns.to_list()):
            ax.vlines(
                model,  # loc[i],
                ymin=min([0, df_bias[model][0]]),
                ymax=max([0, df_bias[model][0]]),
                color='black',
            )
        ax.plot(df_bias.columns, df_bias.iloc[0], "o", color='black')

        # Correlation lollipop plot
        df_corr = pd.DataFrame(
            data=[[average(stats[o]['PearsonR'])
                  for o in range(len(objects))]],
            columns=[o.model for o in objects],
        )
        for i, model in enumerate(df_corr.columns.to_list()):
            ax2.vlines(
                model,  # loc[i],
                ymin=min([0, df_corr[model][0]]),
                ymax=max([0, df_corr[model][0]]),
                linestyle='--',
                color='darkgray',
            )
        ax2.plot(df_corr.columns, df_corr.iloc[0], "x", color='dimgrey')

        # observations horizontal line
        if mean_obs:
            for s, stat in enumerate(stats):
                ax.hlines(
                    stat['obs_mean'].mean(), s-0.5, s+0.5,
                    linestyles=(0, (3, 10, 1, 10)),
                    label=tr.summary_bar_chart['mean_label'][LANG],
                    color='black',
                )

        # abscissa line
        ax.hlines(0, -1, len(objects), color='black', linewidth=1)

        if groups_labels is not None:
            ax.set_title(groups_labels[list(axs).index(ax)])
        ax2.set_ylim(0, 1)

        ax.grid(False, which='both')
        ax2.grid(False, which='both')

    # Get more space and align axes
    axs[-1].set_ylim(axs[-1].get_ylim()[0], axs[-1].get_ylim()[1]*1.2)
    for ax2 in axs2:
        _align_yaxis(axs[-1], 0, ax2, 0, 1)
    if mean_obs:
        axs[0].set_ylabel(tr.summary_bar_chart['ylabel_obs'][LANG])
    else:
        axs[0].set_ylabel(tr.summary_bar_chart['ylabel'][LANG])
    axs2[-1].set_ylabel(tr.summary_bar_chart['corr'][LANG])

    # customize subplots
    for ax, ax2 in zip(axs, axs2):
        if ax is not axs[0]:
            ax.get_yaxis().set_visible(False)
        if ax is not axs[-1]:
            ax2.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        if ax is not axs[0]:
            ax.spines['left'].set_visible(False)
            ax2.spines['left'].set_visible(False)
        if ax is not axs[-1]:
            ax.spines['right'].set_visible(False)
            ax2.spines['right'].set_visible(False)

    # defining legend
    lines = [mlines.Line2D([], [], color='black', marker='o', linewidth=0),
             mlines.Line2D([], [], marker='x', color='dimgrey',
                           linestyle='--', linewidth=1)]
    lines_labs = [tr.summary_bar_chart['bias'][LANG],
                  tr.summary_bar_chart['corr'][LANG]]
    if mean_obs:
        lines.append(mlines.Line2D([], [], color='black', linestyle='dashdot'))
        lines_labs.append(tr.summary_bar_chart['mean_label'][LANG])
    axs[0].legend(
        lines,
        lines_labs,
        loc='upper left',
        # loc='lower left',
        # bbox_to_anchor=(0, 1.01),
    )
    main_ax.legend(
        handles=[
            mlines.Line2D([], [], color=c, label=l, linewidth=3)
            for c, l in zip(colors_legend, labels_legend)
        ],
        title='RMSE',
        ncol=ncol,
        loc='upper right',
        # loc='lower right',
        # bbox_to_anchor=(1, 1.02),
    )
    main_ax.set_title(title)

    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)

    # save figure
    _mpl.save_figure(output_file, file_formats, bbox_inches='tight')

    return fig, main_ax


plot_bb = plot_summary_bar_chart
IMPLEMENTED_PLOTS['summary_bar_chart'] = plot_summary_bar_chart


def plot_exceedances_scores(
        objects, threshold, score_list=None, forecast_day=0, title="",
        labels=None, colors=None, subregions=None, subregion_labels=None,
        file_formats=['png'], output_file=None, output_csv=None, bar_kwargs={},
        start_end=None):
    """
    Barplot for scores.

    Draw a barplot showing thirteen scores, with one bar per object.
    If there are subregions, thirteen barplots are built with one set of bars
    per region.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecast_day : int
        Integer corresponding to the chosen forecast day used for plotting.
    threshold : scalar
        Threshold value used to compute the scores.
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

    title : str
        Title for the plot. If subregions is not None, it can contain
        '{score}'.
    labels : None or list of str
        List of labels for the legend corresponding to the objects.
    colors : None or list of str
        Bar colors corresponding to each object.
    subregions : None or list of list of str
        One set of bars per sub-region will be drawn. The sub-regions must be
        given like [['FR'], ['ES', 'FR'], 'all', ...], where 'FR', 'ES', ...
        are the first letters of the station codes you want to keep
        and 'all' means that all stations are kept.
    subregion_labels : None or list of str
        List of labels for the xticks. These labels corresponds to sub-region
        define *subregions* argument. Labels can contain '{nbStations}' that
        will be replaced by the corresponding number of stations used to
        compute the score (warning: if several objects are displayed, the
        number of stations corresponds to the first one). This argument
        is ignore if subregions is None.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window. If subregions is not None, it must
        contain '{score}' instead of the score name.
    output_csv : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.
    file_formats : list of str
        List of file extensions.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    Returns
    -------
    List of couples (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        Figure and Axes objects corresponding to each plots. Note that if the
        plots have been shown in the user interface window, these figures and
        axes will not be usable again.

    """
    # check if objects are defined on the same period and same stations
    _check_period(objects, False)
    _check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, series_type = \
        _default_params(objects, labels, colors, None, None)

    figs = []
    plt.close('all')

    if subregions is None:
        # stats computing
        stats = evt.tables.exceedances_scores(
            objects, forecast_day, thresholds=[threshold], title=None,
            output_file=output_csv, labels=labels, float_format=None,
            start_end=start_end, score_list=score_list)[0]
        fig = plt.figure()
        ax = plt.gca()
        stats.T.plot.bar(
            ax=ax,
            color=colors,
            legend=True,
            title=title,
            **bar_kwargs
        )
        plt.xticks(rotation=45, ha='right')

        # save figure
        _mpl.save_figure(output_file, file_formats)

        # save in CSV format
        if output_csv is not None:
            stats.T.to_csv(path_or_buf=output_csv, sep=" ",
                           na_rep="nan")

        figs.append((plt.gcf(), plt.gca()))

    else:
        stats = []
        if subregion_labels is None:
            subregion_labels = [str(region) for region in subregions]
        for region in subregions:
            if region == 'all':
                tmp_objs = objects
            else:
                tmp_objs = [obj.select_countries(region, inplace=False)
                            for obj in objects]
            stats.append(evt.tables.exceedances_scores(
                tmp_objs, forecast_day, thresholds=[threshold], title=None,
                output_file=output_csv, labels=labels, float_format=None,
                start_end=start_end, score_list=score_list)[0])
        for sc in stats[0].columns:
            df = pd.DataFrame(
                {lab: st[sc] for lab, st in zip(subregion_labels, stats)}
            )
            fig = plt.figure(constrained_layout=True)
            ax = plt.gca()
            df.T.plot.bar(
                ax=ax,
                color=colors,
                legend=True,
                title=title.format(score=sc),
                **bar_kwargs
            )
            plt.xticks(rotation=45, ha='right')

            # save figure
            figs.append((fig, ax))
            if output_file is not None:
                if "{score}" not in output_file:
                    raise evt.EvaltoolsError(
                        "'{score}' must be contained " +
                        "in the name of the output file !!!")
            _mpl.save_figure(output_file.format(score=sc), file_formats)

            # save in CSV format
            if output_csv is not None:
                if "{score}" not in output_csv:
                    raise evt.EvaltoolsError(
                        "'{score}' must be contained " +
                        "in the name of the output csv file !!!")
                df.T.to_csv(
                    path_or_buf=output_csv.format(score=sc),
                    sep=" ",
                    na_rep="nan",
                )

    return figs
