# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers scores plotting functions."""

import datetime as dt
import numpy as np
import pandas as pd
from functools import reduce
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import evaltools as evt
import evaltools.translation as tr
from evaltools._deprecate import deprecate_func, deprecate
from . import plotting as pl


def _save_figure(outputFile, file_formats, bbox_inches=None, **kwargs):
    """
    Save the figure in different formats or show it if outputFile is None.

    Parameters
    ----------
    outputFile : None or str
        Path of the file where to save the plots (without extension).
    file_formats : list of str
        List of file extensions among 'svgz', 'ps', 'emf', 'rgba', 'raw',
        'pdf', 'svg', 'eps' or 'png'.
    **kwargs :
        Keys of these arguments must be contained in the name of the output
        file and will be replaced by the corresponding values.

    """
    if outputFile is None:
        if pl.USER_INTERFACE_WINDOW:
            plt.show()
    else:
        for key in kwargs.keys():
            if "{" + key + "}" not in outputFile:
                raise evt.EvaltoolsError(
                    "'{" + key + "}' must be " +
                    "contained in the name of the output file !!!")
        for ext in file_formats:
            plt.savefig(
                (outputFile+".{ext}").format(ext=ext, **kwargs),
                bbox_inches=bbox_inches,
            )


def _is_outlier(points, thresh=3.5):
    """
    Look for outliers.

    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters
    ----------
    points : An numobservations by numdimensions array of observations
    thresh : The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    Returns
    -------
        A numobservations-length boolean array.

    References
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    """
    idx_all_nan = np.isnan(points).all(axis=1)
    if len(points.shape) == 1:
        points = points[:, None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "All-NaN slice encountered")
        median = np.nanmedian(points, axis=0)
    diff = np.nansum((points - median)**2, axis=-1)
    diff[idx_all_nan] = np.nan
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745*diff/med_abs_deviation
    # print(modified_z_score)

    return modified_z_score > thresh


def _black_axes(ax):
    """Paint in black y=0 and x=0."""
    if ax.axes.get_xlim()[0] < 0 and ax.axes.get_ylim()[1] > 0:
        plt.axvline(x=0, color='k')
    if ax.axes.get_ylim()[0] < 0 and ax.axes.get_ylim()[1] > 0:
        plt.axhline(y=0, color='k')


def _set_minor_ticks(nb_x, nb_y):
    """Define number of minor ticks between each major one."""
    ax = plt.gca()
    minorLocator = AutoMinorLocator(n=nb_x)
    ax.xaxis.set_minor_locator(minorLocator)
    minorLocator = AutoMinorLocator(n=nb_y)
    ax.yaxis.set_minor_locator(minorLocator)


class _Curves(object):
    """
    Class designed to collect and work on the different curves in a chart.

    Parameters
    ----------
    x : 1D array-like
        Values for the x axis.

    """

    def __init__(self, x, ax=None, constrained_layout=True):
        # abscissa
        self.x = x
        # curves
        self.style = {}
        self.curves = pd.DataFrame(index=self.x)
        self.curves.index.name = 'X'
        if ax is None:
            self.fig = plt.figure(
                constrained_layout=constrained_layout,
            )
            self.ax = self.fig.add_subplot(1, 1, 1)
        else:
            self.ax = ax
            self.fig = ax.figure
        # draw grid
        self.ax.grid(which='both', ls='-', c='#B0B0B0')
        # try to control number of yticks
        self.ax.locator_params(axis='y', nbins=11, min_n_ticks=10)

    def add_curve(self, key, y, style={}):
        """Add a curve."""
        self.curves[key] = y
        self.style[key] = style

    def plot(self, black_axes=False, outlier_thresh=None, legend=True,
             legend_type='fig'):
        """
        Plot curves.

        Parameters
        ----------
        black_axes : bool
            If true, y=0 and x=0 lines are painted in black.
        outlier_thresh : scalar or None
            If not None, it correspond to the threshold used in
            evaltools.plotting._is_outlier to determine if a model is an
            outlier. If outliers are detected, y boudaries do not take
            them into account.
        legend_type : str
            Set to 'fig' for figure legend or to 'ax' for axis legend.

        """
        for key in self.curves.columns:
            plt.plot(self.x, self.curves[key], **self.style[key])

        if legend is True:
            # choose nb of columns in legend
            if len(self.curves.columns) > 4:
                legend_ncol = 2
            else:
                legend_ncol = 1
            if legend_type == 'fig':
                plt.figlegend(ncol=legend_ncol)
            elif legend_type == 'ax':
                plt.legend(ncol=legend_ncol)

        # paint in black y=0 and x=0
        if black_axes is True:
            _black_axes(self.ax)
        # set y boudaries without taking outliers into account
        if outlier_thresh is not None:
            self.ignore_outliers(outlier_thresh)

    def plot_median(self, style={}, envelope=False, min_val=1):
        """
        Plot median of curves.

        Parameters
        ----------
        style : dict
            Keyword arguments passed to matplotlib.pyplot.plot function.
        envelope : bool
            If True, draw quartiles one and three around the median curve .
        min_val : int
            Minimal number of values required to compute the median.

        """
        quartiles = self.curves.quantile([0.25, 0.5, 0.75], axis=1).T
        idx_not_enough = (~self.curves.isna()).sum(axis=1) < min_val
        quartiles.loc[idx_not_enough] = np.nan
        if envelope is True:
            if 'linestyle' in style:
                style.pop('linestyle')
            plt.plot(self.x, quartiles[0.5], linestyle='-', **style)
            plt.plot(self.x, quartiles[[0.25, 0.75]],
                     linestyle='--', **style)
        else:
            plt.plot(self.x, quartiles[0.5], **style)

    def plot_mean(self, style={}, envelope=False, min_val=1):
        """
        Plot mean of curves.

        Parameters
        ----------
        style : dict
            Keyword arguments passed to matplotlib.pyplot.plot function.
        envelope : bool
            If True, draw +/- the standard deviation around the mean curve.
        min_val : int
            Minimal number of values required to compute the mean.

        """
        mean = self.curves.mean(axis=1).T
        idx_not_enough = (~self.curves.isna()).sum(axis=1) < min_val
        mean.loc[idx_not_enough] = np.nan
        if envelope is True:
            if 'linestyle' in style:
                style.pop('linestyle')
            plt.plot(self.x, mean, linestyle='-', **style)
            std = self.curves.std(axis=1).T
            plt.plot(self.x, mean + std, linestyle='--', **style)
            plt.plot(self.x, mean - std, linestyle='--', **style)
        else:
            plt.plot(self.x, mean, **style)

    def ignore_outliers(self, outlier_thresh):
        """Set y boudaries without taking outliers into account."""
        # seek outliers
        points = self.curves.values.T
        outliers = _is_outlier(points, outlier_thresh)
        # find min/max
        ymin = np.nanmin(points[~outliers, :])
        ymax = np.nanmax(points[~outliers, :])
        margin = (ymax - ymin)*.05
        self.ax.set_ylim(bottom=ymin-margin, top=ymax+margin)

    def write_csv(self, outputCSV, **kwargs):
        """
        Save data.

        Parameters
        ----------
        outputCSV : str or None
            File where to save the data. The file name must contain {score}
            instead of the score name.
        **kwargs :
            Keys of these arguments must be contained in the name of the
            output file and will be replaced by the corresponding values.

        """
        # check compulsory keys in file name
        for key in kwargs.keys():
            if "{" + key + "}" not in outputCSV:
                raise evt.EvaltoolsError(
                    "'{" + key + "}' must be " +
                    "contained in the name of the output csv file !!!")

        self.curves.to_csv(outputCSV.format(**kwargs), sep=' ', na_rep='nan',
                           float_format='%g', header=True, index=True)

    def set_time_xaxis(self, xticking, date_format=None):
        """Set pyplot x axis as a datetime axis."""
        if date_format is not None:
            self.ax.xaxis.set_major_formatter(
                mdates.DateFormatter(date_format))
        if xticking != 'auto':
            if xticking == 'daily':
                loc = mdates.WeekdayLocator(byweekday=range(7))
            elif xticking == 'mondays':
                loc = mdates.WeekdayLocator(byweekday=0)
            elif xticking == 'bimonthly':
                loc = mdates.WeekdayLocator(byweekday=1, interval=2)
            else:
                raise evt.EvaltoolsError(
                    "xticking argument must be either 'daily', 'mondays' " +
                    "'bimonthly' or 'auto'")
            self.ax.xaxis.set_major_locator(loc)
        self.fig.autofmt_xdate(rotation=45, ha='right')
        plt.tick_params(axis='x', which='major', labelsize='small')


def _mscatter(x, y, ax=None, m=None, **kw):
    """Make a scatter plot with a different marker for each point."""
    import matplotlib.markers as mmarkers

    # get pyplot axis
    if not ax:
        ax = plt.gca()

    # classic scatter plot
    sc = ax.scatter(x, y, **kw)

    # modify marker style for each point
    if (m is not None) and (len(m) == len(x)):
        if 'c' in kw:
            if pd.Series(kw['c']).isna().any():
                raise evt.EvaltoolsError(
                    "_scatter function can not manage nan values in c " +
                    "argument.")
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


@deprecate_func('plot_meanTimeScores', pl.plot_mean_time_scores)
def plot_meanTimeScores(
        objects, score_list, labels=None, colors=None,
        linestyles=None, markers=None, outputFile=None, suptitle="", title="",
        score_names=None, file_formats=['png'], min_nb_sta=10,
        availability_ratio=0.75, outputCSV=None, annotation=None,
        xlabel=None, black_axes=False, outlier_thresh=None,
        nb_of_minor_ticks=(1, 2)):
    """
    Plot the mean of time scores for each forecast time.

    This function is based on Evaluator.meanTimeScores method.
    Scores are first computed for each time along available stations. Then,
    the mean of these scores is taken for each forecast time.

    On each plot, there will be as many lines as there are objects given in
    the object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score_list : list of str
        List of computed scores.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    outputFile : None or str
        File where to save the plots (without extension). The file name
        must contain {score} instead of the score name. If None, the figure
        is shown in a popping window.
    suptitle : str
        Suptitle for the plots.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    score_names : dict or None
        Dictionary with keys containing the elements of score_list and with
        values the strings to in the title instead of the default scores
        names. If None, score names are written as in score_list.
    file_formats : list of str
        List of file extensions.
    min_nb_sta : int
        Minimal number of station values available in both obs and
        sim required per datetime to compute the scores (before applying
        mean for each forecast time).
    availability_ratio : float
        Minimal rate of data (computed scores per time) available on the
        period required per forecast time to compute the mean scores.
    outputCSV : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    xlabel : str
        Label for the x axis.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.

    Returns
    -------
    List of couples (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        Figure and Axes objects corresponding to each plots. Note that if the
        plots have been shown in the user interface window, these figures and
        axes will not be usable again.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)

    # define default score names
    if score_names is None:
        score_names = dict(zip(score_list, score_list))
    if xlabel is None:
        xlabel = tr.mean_time_scores['xlabel'][pl.LANG]

    # abcissa
    x_max = 24*objects[0].forecastHorizon
    X = range(0, x_max, objects[0].step)

    # scores computing
    stats = []
    for obj in objects:
        stats.append(obj.meanTimeScores(
            score_list,
            min_nb_sta=min_nb_sta,
            availability_ratio=availability_ratio))

    # plotting
    plt.close('all')
    figs = []
    for sc in score_list:
        curves = _Curves(X)
        for Y, l, c, ls, m in zip(stats, labels, colors, linestyles, markers):
            curves.add_curve(l, Y[sc][:len(X)],
                             style={'label': l, 'color': c,
                                    'linestyle': ls, 'marker': m})
        curves.plot(black_axes=black_axes, outlier_thresh=outlier_thresh)
        plt.xlabel(xlabel)
        plt.ylabel(score_names[sc])
        plt.suptitle(suptitle, fontweight='bold', x=0.25, y=0.98)
        plt.title(title.format(score=score_names[sc]), loc='left', x=0.05)
        plt.xlim(left=X[0], right=X[-1])
        major_ticks = np.arange(0, x_max, 12)
        minor_ticks = np.arange(0, x_max, 3)
        curves.ax.set_xticks(major_ticks)
        curves.ax.set_xticks(minor_ticks, minor=True)

        _set_minor_ticks(*nb_of_minor_ticks)

        if annotation is not None:
            plt.annotate(annotation,
                         xy=(0, 1), xycoords='figure fraction', va='top',
                         fontsize='small', fontstyle='italic', color='#7A7A7A')

        # save figure
        figs.append((plt.gcf(), plt.gca()))
        _save_figure(outputFile, file_formats, score=sc)

        # save data
        if outputCSV is not None:
            curves.write_csv(outputCSV.format(score=sc))

    return figs


@deprecate_func('plot_medianStationScores', pl.plot_median_station_scores)
def plot_medianStationScores(
        objects, score_list, labels=None, colors=None, linestyles=None,
        markers=None, outputFile=None, suptitle="", title="",
        score_names=None, file_formats=['png'], availability_ratio=0.75,
        min_nb_sta=10, outputCSV=None, xlabel=None,
        black_axes=False, outlier_thresh=None, nb_of_minor_ticks=(1, 2),
        annotation=None):
    """
    Plot the median of station scores for each forecast time.

    This function is based on Evaluator.medianStationScores method.
    Scores are first computed for each station at every times. Then,
    the median of these scores is taken for each forecast time.

    On each plots, there will be as many lines as there are objects given in
    the object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score_list : list of str
        List of computed scores.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    outputFile : None or str
        File where to save the plots (without extension). The file name
        must contain {score} instead of the score name. If None, the figure
        is shown in a popping window.
    suptitle : str
        Suptitle for the plot.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    score_names : dict
        Dictionary with keys containing the elements of score_list and as
        values the strings writen in the title instead of the default scores
        names.
    file_formats : list of str
        List of file extensions.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        time to compute the scores for each station.
    min_nb_sta : int
        Minimal number of stations required to compute the median of
        the scores.
    outputCSV : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.
    xlabel : str
        Label for the x axis.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        evaltools.plotting._is_outlier to determine if a model is an outlier.
        If outliers are detected, y boudaries do not take them into account.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    List of couples (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        Figure and Axes objects corresponding to each plots. Note that if the
        plots have been shown in the user interface window, these figures and
        axes will not be usable again.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)
    if xlabel is None:
        xlabel = tr.mean_time_scores['xlabel'][pl.LANG]

    # define default score names
    if score_names is None:
        score_names = dict(zip(score_list, score_list))

    # abcissa
    if objects[0].seriesType == 'hourly':
        x_max = 24*objects[0].forecastHorizon
        X = range(0, x_max, objects[0].step)
    elif objects[0].seriesType == 'daily':
        X = range(objects[0].forecastHorizon)

    # scores computing
    stats = []
    for obj in objects:
        stats.append(obj.medianStationScores(
            score_list,
            availability_ratio=availability_ratio,
            min_nb_sta=min_nb_sta))

    # plotting
    plt.close('all')
    figs = []
    for sc in score_list:
        curves = _Curves(X)
        for Y, l, c, ls, m in zip(stats, labels, colors, linestyles, markers):
            curves.add_curve(l, Y[sc][:len(X)],
                             style={'label': l, 'color': c,
                                    'linestyle': ls, 'marker': m})
        curves.plot(black_axes=black_axes)
        plt.xlabel(xlabel)
        plt.ylabel(score_names[sc])
        plt.suptitle(suptitle, fontweight='bold', x=0.25, y=0.98)
        plt.title(title.format(score=score_names[sc]), loc='left', x=0.05)
        plt.xlim(left=X[0], right=X[-1])
        if objects[0].seriesType == 'hourly':
            major_ticks = np.arange(0, x_max, 12)
            minor_ticks = np.arange(0, x_max, 3)
            curves.ax.set_xticks(major_ticks)
            curves.ax.set_xticks(minor_ticks, minor=True)
        elif objects[0].seriesType == 'daily':
            plt.xticks(X, ['D{}'.format(d) for d in X])

        _set_minor_ticks(*nb_of_minor_ticks)

        if annotation is not None:
            plt.annotate(annotation,
                         xy=(0, 1), xycoords='figure fraction', va='top',
                         fontsize='small', fontstyle='italic', color='#7A7A7A')

        # save figure
        figs.append((plt.gcf(), plt.gca()))
        _save_figure(outputFile, file_formats, score=sc)

        # save data
        if outputCSV is not None:
            curves.write_csv(outputCSV.format(score=sc))

    return figs


@deprecate_func('plot_timeScores', pl.plot_time_scores)
def plot_timeScores(
        objects, score_list, term, labels=None, colors=None,
        linestyles=None, markers=None, outputFile=None, suptitle="", title="",
        score_names=None, file_formats=['png'], min_nb_sta=10, outputCSV=None,
        xticking='auto', date_format='%Y-%m-%d', black_axes=False,
        outlier_thresh=None, nb_of_minor_ticks=(2, 2), annotation=None,
        hourly_timeseries=False, start_end=None):
    """
    Plot time scores at a chosen time for each forecast day.

    This function is based on Evaluator.timeScores method.
    On each plots, there will be as many lines as there are objects the
    object list.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score_list : list of str
        List of computed scores.
    term : int.
        Integer corresponding to the chosen term used for plotting.
        If the series type of the objects is 'hourly', it refers
        to the forecast time (for example between 0 and 95 if the forecast
        horizon is 4 days). If the series type of the objects is 'daily',
        it refers to the forcast day (for example between 0 and 3 if the
        forecast horizon is 4 days).
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    outputFile : None or str
        File where to save the plots (without extension). The file name
        must contain {score} instead of the score name. If None, the figure
        is shown in a popping window.
    suptitle : str
        Suptitle for the plots.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    score_names : dict
        Dictionary with keys containing the elements of score_list and as
        values the strings writen in the title instead of the default scores
        names.
    file_formats : list of str
        List of file extensions.
    min_nb_sta : int
        Minimal number of station values available in both obs and sim
        required per datetime to compute the scores.
    outputCSV : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.
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
    annotation : str or None
        Additional information to write in figure's upper left corner.
    hourly_timeseries : bool
        If True, every time step are plotted. In this case, argument term
        refers to the forecast day. This argument is ignored if the series
        type of the objects is 'daily'.
    start_end : None or list of two datetime.date objects
        Boundary dates for abscissa.

    Returns
    -------
    List of couples (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        Figure and Axes objects corresponding to each plots. Note that if the
        plots have been shown in the user interface window, these figures and
        axes will not be usable again.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)

    # define whether if term is the forecast day or forecast time
    if seriesType == 'hourly':
        if hourly_timeseries:
            day = term
        else:
            # forecast day corresponding to term
            day = term//24
            # hour of the day corresponding term
            t = term % 24
    elif seriesType == 'daily':
        day = term

    # define default score names
    if score_names is None:
        score_names = dict(zip(score_list, score_list))

    # scores computing
    stats = []
    for obj in objects:
        Y = obj.timeScores(score_list,
                           min_nb_sta=min_nb_sta)['D{day}'.format(day=day)]
        if seriesType == 'hourly' and not hourly_timeseries:
            idx = Y.index.time == dt.time(t, 0)
            Y = Y[idx]
        stats.append(Y)

    # plotting
    plt.close('all')
    figs = []
    for sc in score_list:
        if seriesType == 'hourly':
            Xticks = np.unique(stats[0].index.date)
            if hourly_timeseries:
                X = Y.index
            else:
                X = [dt.datetime.combine(x, dt.time(t)) for x in Xticks]
            Xticks = np.append(Xticks, Xticks[-1]+dt.timedelta(days=1))
        elif seriesType == 'daily':
            Xticks = stats[0].index
            X = Xticks
        curves = _Curves(X, constrained_layout=False)

        for Y, l, c, ls, m, obj in zip(stats, labels, colors, linestyles,
                                       markers, objects):
            if Y[sc].isna().all():
                print("*** all nan values for {sc} of {label}".format(
                    sc=sc, label=l))
            else:
                curves.add_curve(l, Y[sc][:len(X)],
                                 style={'label': l, 'color': c,
                                        'linestyle': ls, 'marker': m})
        curves.plot(black_axes=black_axes, outlier_thresh=outlier_thresh)

        plt.ylabel(score_names[sc])
        plt.suptitle(suptitle, fontweight='bold', x=0.25, y=0.98)
        plt.title(title.format(score=score_names[sc]), loc='left', x=0.05)

        # set xticks
        curves.set_time_xaxis(xticking, date_format)
        if start_end is None:
            plt.xlim(left=Xticks[0], right=Xticks[-1])
        else:
            plt.xlim(left=start_end[0], right=start_end[1])
        _set_minor_ticks(*nb_of_minor_ticks)

        if annotation is not None:
            plt.annotate(annotation,
                         xy=(0, 1), xycoords='figure fraction', va='top',
                         fontsize='small', fontstyle='italic', color='#7A7A7A')

        # save figure
        plt.tight_layout()
        figs.append((plt.gcf(), plt.gca()))
        _save_figure(outputFile, file_formats, score=sc)

        # save data
        if outputCSV is not None:
            curves.write_csv(outputCSV.format(score=sc))

    return figs


@deprecate_func('plot_quarterlyMedianScore', pl.plot_quarterly_score)
def plot_quarterlyMedianScore(
        files, labels, colors, first_quarter=None, last_quarter=None,
        score='RMSE', outputFile=None, linestyles=None, markers=None,
        suptitle="", title="", thres=None, file_formats=['png'], ylabel=None,
        origin_zero=False, black_axes=False):
    """
    Plot quarterly values saved in given files.

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
    colors : list of str
        Line colors corresponding to each files.
    first_quarter : str
        String corresponding to the oldest plotted quarter.
    last_quarter : str
        String corresponding to the latest plotted quarter.
    score : str
        The score that should correspond to data stored in the files.
    outputFile : str
        File where to save the plot (without extension).
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    suptitle : str
        Suptitle for the plots.
    title : str
        Title for the plots.
    thres : None or float
        If not None, a horizontal yellow line is plotted with
        equation y = thres.
    file_formats : list of str
        List of file extensions.
    ylabel : str
        Ordinate axis label.
    origin_zero : bool
        If True, minimal value of y axis is set to zero.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # define linestyles in case of None
    if linestyles is None:
        linestyles = ['-']*len(files)

    # define markers in case of None
    if markers is None:
        markers = ['.']*len(files)

    # find first and last quarters in case of None
    if last_quarter is None or first_quarter is None:
        dfs = [pd.read_csv(f, sep=' ', index_col=0) for f in files]
        quarter_list = []
        for df in dfs:
            quarter_list.extend([evt.quarter.Quarter.from_string(q)
                                 for q in df.index])
        quarter_list.sort()
        if first_quarter is None:
            first_quarter = quarter_list[0].string
        if last_quarter is None:
            last_quarter = quarter_list[-1].string

    end_quarter = evt.quarter.Quarter.from_string(last_quarter)
    start_quarter = evt.quarter.Quarter.from_string(first_quarter)
    all_quarters = end_quarter.range(start_quarter)

    Ys = {}
    for kpiFile, lab in zip(files, labels):
        df = pd.read_csv(kpiFile, sep=' ', index_col=0)
        Ys[lab] = pd.DataFrame(index=[q.__repr__() for q in all_quarters],
                               columns=[score])
        Ys[lab].update(df)

    X = range(len(all_quarters))

    plt.close('all')
    fig = plt.figure()
    if thres is not None:
        plt.plot(X, [thres]*len(X), color='#FFA600')
    for lab, c, ls, m in zip(labels, colors, linestyles, markers):
        Y = Ys[lab].astype('float')
        plt.plot(X, Y, color=c, linestyle=ls, marker=m, label=lab)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(Ys[labels[0]].keys()[0])
    plt.suptitle(suptitle, fontweight='bold', x=0.25, y=0.98)
    plt.title(title, loc='left', x=0.05)
    plt.grid(True, ls='-', c='#B0B0B0')
    plt.figlegend()
    plt.xlim(left=X[0], right=X[-1])
    if origin_zero is True:
        plt.ylim(bottom=0)
    plt.xticks(X, Ys[lab].index)
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)
    fig.autofmt_xdate(bottom=0.2, rotation=80, ha='right')

    # paint in black y=0 and x=0
    if black_axes is True:
        _black_axes(fig.get_axes()[0])

    # save figure
    plt.tight_layout()
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_stationScores', pl.plot_station_scores)
def plot_stationScores(
        obj, score, forecastDay=0, outputFile=None, title="",
        bbox=[-26, 46, 28, 72], file_formats=['png'], point_size=5,
        higher_above=True, order_by=None, availability_ratio=0.75, vmin=None,
        vmax=None, cmap=cm.jet, rivers=False, outputCSV=None, interp2D=False,
        sea_mask=False, land_mask=False, boundary_resolution='50m',
        cmaplabel='', extend='neither', land_color='none', sea_color='none',
        marker='o', mark_by=None, boundaries_above=False, bbox_inches='tight',
        grid_resolution=None):
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
    forecastDay : int.
        Integer corresponding to the chosen forecastDay used for plotting.
    outputFile : str or None
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
    cmap : matplotlib.colors.Colormap object
        Colors used for plotting (default: matplotlib.cm.jet).
    rivers : bool
        If True, rivers and lakes are drawn.
    outputCSV : str or None
        File where to save the data. The file name can contain {score}
        instead of the score name.
    interp2D : bool
        If True, a 2D linear interplation is performed on scores values.
    sea_mask : bool
        If True, scores ought to be drawn over sea are masked.
    land_mask : bool
        If True, scores ought to be drawn over land are masked.
    boundary_resolution : str
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
        If 'tight', try to figure out the tight bbox of the figure (in this
        case, evltools.plotting.figsize is no longer respected).
    grid_resolution : couple of scalars
        Couple of scalars corresponding to meridians and parallels spacing
        in degrees.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
    except ImportError as e:
        print("Module cartopy is required to use plot_stationScores function.")
        return

    if not np.isin(['lat', 'lon'], obj.stations.columns).all():
        print(("lat/lon coordinates not found in {}.stations. Station " +
               "scores map can not be plotted.").format(obj))
        return

    # compute scores
    df = obj.stationScores(
        [score],
        availability_ratio=availability_ratio)['D{}'.format(forecastDay)]

    # sort values by plotted values (or metadata if order_by is not None)
    df.sort_values(by=score, ascending=higher_above, inplace=True)
    if order_by is not None:
        if np.in1d(order_by, obj.stations.columns):
            df = df.merge(obj.stations[[order_by]], how='left',
                          left_index=True, right_index=True, sort=False)
            df.sort_values(ascending=higher_above, by=order_by,
                           inplace=True)
        else:
            print(("*** Values can't be ordered by {} since " +
                   "this variable is not in the object's " +
                   "metadata").format(order_by))
    df = df[score]

    # save data
    if outputCSV is not None:
        csv_data = obj.stations[['lat', 'lon']].loc[df.index]
        csv_data[score] = df
        csv_data.index.name = 'station'
        csv_data.sort_values(by=score)
        csv_data.to_csv(outputCSV.format(score=score), sep=' ',
                        na_rep='nan', float_format='%g', header=True,
                        index=True)

    # plotting
    plt.close('all')
    fig = plt.figure()

    ax = plt.axes(projection=ccrs.PlateCarree())

    # map layout
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 'land',
            boundary_resolution,
            edgecolor='none',
            facecolor=land_color,
        )
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 'ocean',
            boundary_resolution,
            edgecolor='none',
            facecolor=sea_color,
        )
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 'land',
            boundary_resolution,
            edgecolor='k',
            facecolor='none',
            zorder=2 + 10*boundaries_above,
        )
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 'ocean',
            boundary_resolution,
            edgecolor='k',
            facecolor='none',
            zorder=2 + 10*boundaries_above,
        )
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land',
            scale=boundary_resolution,
            edgecolor='grey',
            facecolor='none',
            zorder=2 + 10*boundaries_above,
        )
    )

    # land/sea mask
    if land_mask is True:
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land',
                                                    boundary_resolution,
                                                    edgecolor='k',
                                                    facecolor='w'),
                       zorder=11)
    if sea_mask is True:
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean',
                                                    boundary_resolution,
                                                    edgecolor='k',
                                                    facecolor='w'),
                       zorder=11)

    if rivers is True:
        ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m',
            edgecolor='face', facecolor=np.array((152, 183, 226)) / 256.))
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                'physical', 'rivers_lake_centerlines', '50m',
                edgecolor=np.array((152, 183, 226)) / 256.,
                facecolor='none'),
            alpha=0.5)

    # set boundary box
    x1, x2, y1, y2 = bbox
    ax.set_extent(bbox, crs=ccrs.PlateCarree())

    # title
    plt.title(title)

    # annotate number of stations
    annotation = (tr.station_scores['annotation'][pl.LANG]).format(
                        av=sum(df.notna()),
                        total=df.shape[0],
                        mini=round(df.min(), 2),
                        avg=round(df.mean(), 2),
                        maxi=round(df.max(), 2))
    plt.annotate(annotation,
                 xy=(0, 0),
                 xytext=(0, -10),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 va='top',
                 fontsize='small', fontstyle='italic', color='#7A7A7A')

    # custom axis for the colorbar
    divider = make_axes_locatable(ax)

    if interp2D is True:
        from scipy.interpolate import griddata
        grid_lat, grid_lon = np.mgrid[y1:y2:1000j, x1:x2:1000j]
        interp_data = griddata(
            np.array(obj.stations[['lat', 'lon']].loc[df.index]),
            df.values, (grid_lat, grid_lon),
            method='linear')
        im = ax.imshow(
            interp_data, extent=bbox,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            zorder=2,
            origin='lower',
        )
        # im = ax.pcolormesh(grid_lon[0, :], grid_lat[:, 0], interp_data,
        #                    transform=ccrs.Mercator())
        cax = divider.append_axes(
            "right",
            size="3%",
            axes_class=plt.Axes,
            pad=0.2,
        )
        plt.colorbar(im, cax=cax, extend=extend, label=cmaplabel)
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
        scatter = _mscatter(
            x, y, ax=ax, m=markers, marker=marker, s=point_size, c=df,
            cmap=cmap, edgecolors=None, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), zorder=2,
        )
        cax = divider.append_axes(
            "right",
            size="3%",
            axes_class=plt.Axes,
            pad=0.2,
        )
        plt.colorbar(scatter, cax=cax, extend=extend, label=cmaplabel)

    # grid
    if grid_resolution is not None:
        meridians = np.concatenate((
            np.arange(0, -200, -grid_resolution[0]),
            np.arange(0, 200, grid_resolution[0]),
        ))
        parallels = np.concatenate((
            np.arange(0, -100, -grid_resolution[1]),
            np.arange(0, 100, grid_resolution[1]),
        ))
        gl = ax.gridlines(
            xlocs=meridians,
            ylocs=parallels,
            draw_labels=True,
            linewidth=1,
            color='k',
            linestyle='--',
            zorder=3,
        )
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    # save figure
    _save_figure(outputFile, file_formats, bbox_inches=bbox_inches)

    return fig, ax


@deprecate_func('plot_taylorDiagram', pl.plot_taylor_diagram)
def plot_taylorDiagram(
        objects, forecastDay=0, norm=True, colors=None, markers=None,
        point_size=100, outputFile=None, labels=None, suptitle="", title="",
        file_formats=['png'], threshold=0.75, outputCSV=None, frame=False,
        crmse_levels=10, annotation=None):
    """
    Taylor diagram.

    This function is based on Evaluator.FDscores method.
    Pearson correlation and variance ratio are first computed from all data
    of a choosen forecast day (values for all station at all times are
    considered as a simple 1D array).

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecastDay : int
        Forecast day for which to use data to compute the Taylor diagram.
    norm : bool
        If True, standard deviation and CRMSE are divided by the standard
        deviation of observations.
    colors : None or list of str
        Marker colors corresponding to each objects.
    markers : None or list of str
        Marker shapes corresponding to each objects.
    point_size : float
        Point size (as define in matplotlib.pyplot.scatter).
    outputFile : str
        File where to save the plot (without extension).
    labels : list of str
        List of labels for the legend.
    suptitle : str
        Diagram suptitle.
    title : str
        Diagram title.
    file_formats : list of str
        List of file extensions.
    threshold : int or float
        Minimal number (if type(threshold) is int) or minimal rate
        (if type(threshold) is float) of data available in both obs and
        sim required to compute the scores.
    outputCSV : str or None
        File where to save the data.
    frame : bool
        If false, top and right figure boundaries are not drawn.
    crmse_levels : int
        Number of CRMSE arcs of circle.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    linestyles = None
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(
            objects, labels, colors, linestyles, markers,
            default_marker='^',
        )

    # differences between original diagram and normalized one
    if norm:
        sd_label = tr.taylor_diagram['sdr'][pl.LANG]
        score = 'SDratio'
        refstd = 1
    else:
        sd_label = tr.taylor_diagram['sd'][pl.LANG]
        score = 'sim_std'
        refstd = objects[0].FDscores(score_list=['obs_std'],
                                     forecast_days=[forecastDay],
                                     threshold=threshold).iloc[0, 0]

    # max of x and y axes
    smax = 1.6*refstd

    # plotting
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid(False, which='both')

    ax.set_xlim(0, smax)
    ax.set_ylim(0, smax)
    if norm:
        major_ticks = np.arange(0, smax, 1)
        minor_ticks = np.arange(0, smax, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
    ax.set_aspect('equal')

    # draw CRMSE circles
    if frame is True:
        xs, ys = np.meshgrid(np.linspace(0, smax), np.linspace(0, smax))
        rms = np.sqrt((refstd-xs)**2 + ys**2)
    else:
        xs, ys = np.meshgrid(np.linspace(0, smax, 1000),
                             np.linspace(0, smax, 1000))
        rms = np.where(xs**2+ys**2 < (smax*(15/16.))**2,
                       np.sqrt((refstd-xs)**2 + ys**2),
                       np.nan)
        ax.set_frame_on(False)
        ax.axhline(y=0, xmax=15/16., color='k')
        ax.axvline(x=0, ymax=15/16., color='k')
        ticks = np.array(ax.get_xticks())
        ticks = ticks[ticks <= smax*(15/16.)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    contours = ax.contour(xs, ys, rms, crmse_levels, colors='k',
                          linestyles='dotted', linewidths=0.5)
    if not norm:
        ax.clabel(contours, inline=1, fontsize=10)

    # draw correlation rays and values
    corr_steps = np.concatenate((np.arange(10)/10., [0.95, 0.99]))
    m = 0.02*refstd
    for corr in corr_steps:
        ang = np.arccos(corr)
        x = smax*(15/16.)*np.cos(ang)
        y = smax*(15/16.)*np.sin(ang)
        plt.plot([0, x], [0, y], 'c', lw=0.5)
        if corr != 0.:
            plt.text(x+m*np.cos(ang), y+m*np.sin(ang), corr, color='b',
                     rotation=(ang-np.pi/2)*180/np.pi,
                     verticalalignment='center', horizontalalignment='center')
    plt.text(
        smax*np.sqrt(2)/2, smax*np.sqrt(2)/2,
        tr.taylor_diagram['corr'][pl.LANG], color='b',
        rotation=-45, verticalalignment='center',
        horizontalalignment='center', size='x-large',
    )

    # draw correlation circle
    for r in [refstd, smax*(15/16.)]:
        theta = np.linspace(0, np.pi/2)
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        ax.plot(x, y, c='#FF3FFF', linestyle='-')

    # axes labels
    plt.xlabel(sd_label)
    plt.ylabel(sd_label)

    # plot model points
    csv_data = pd.DataFrame(columns=['corr', score, 'CRMSE'])
    for obj, color, marker, lab in zip(objects, colors, markers, labels):
        df = obj.FDscores(score_list=[score, 'PearsonR', 'CRMSE'],
                          forecast_days=[forecastDay], threshold=threshold)
        corr = df['PearsonR'].loc['D{}'.format(forecastDay)]
        score_value = df[score].loc['D{}'.format(forecastDay)]
        x = corr*score_value
        y = np.sin(np.arccos(corr))*score_value
        ax.scatter(x, y, c=color, marker=marker, label=lab, s=point_size,
                   zorder=10)
        csv_data = pd.concat([csv_data,
                              pd.DataFrame({'corr': corr,
                                            score: score_value,
                                            'CRMSE': df['CRMSE'].loc[
                                                'D{}'.format(forecastDay)]},
                                           index=[lab])],
                             sort=False)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(suptitle, horizontalalignment='center', fontweight='bold')
    plt.title(title)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    _save_figure(outputFile, file_formats, bbox_inches='tight')

    # save data
    if outputCSV is not None:
        csv_data.index.name = 'model'
        csv_data.to_csv(outputCSV, sep=' ', na_rep='nan', float_format='%g',
                        header=True, index=True)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_scoreQuartiles', pl.plot_score_quartiles)
def plot_scoreQuartiles(
        objects, xscore, yscore, colors=None, forecastDay=0, title="",
        outputFile=None, labels=None, availability_ratio=0.75, min_nb_sta=10,
        file_formats=['png'], outputCSV=None, invert_xaxis=False,
        invert_yaxis=False, xmin=None, xmax=None, ymin=None, ymax=None,
        black_axes=False):
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
    colors : None or list of str
        Marker colors corresponding to each objects.
    forecastDay : int
        Forecast day used for score computing.
    title : str
        Chart title.
    outputFile : str
        File where to save the plot (without extension).
    labels : list of str
        List of objects labels for the legend.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    min_nb_sta : int
        Minimal number of stations required to compute the quartiles of
        the scores.
    file_formats : list of str
        List of file extensions.
    outputCSV : str or None
        File where to save the data.
    invert_xaxis, invert_yaxis : bool
        If True, the axis is inverted.
    xmin, xmax, ymin, ymax : None or scalar
        Limits of the axes.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

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
        stats = obj.stationScores(
            score_list=[xscore, yscore],
            availability_ratio=availability_ratio)['D{}'.format(forecastDay)]
        if (np.sum(~stats[xscore].isna(), axis=0) > min_nb_sta):
            quartiles = np.nanpercentile(stats, q=[25, 50, 75], axis=0)
            quartiles = pd.DataFrame(quartiles, index=[25, 50, 75],
                                     columns=stats.columns)
            res[lab] = quartiles
        else:
            res[lab] = pd.DataFrame(np.nan, index=[25, 50, 75],
                                    columns=stats.columns)
            print("Warning: Availability criteria not fulfilled " +
                  "for {} !!!".format(lab))

    # plotting
    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
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
    ax.grid(True)
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
        _black_axes(ax)

    # save figure
    _save_figure(outputFile, file_formats)

    # save data
    if outputCSV is not None:
        with open(outputCSV, 'w') as f:
            for lab in labels:
                f.write(lab+'\n')
                f.write(res[lab].to_string())
                f.write('\n\n')

    return plt.gcf(), plt.gca()


@deprecate_func('plot_timeSeries', pl.plot_time_series)
def plot_timeSeries(
        objects, station_list=None, start_end=None, forecastDay=0,
        colors=None, linestyles=None, markers=None, outputFile=None,
        labels=None, title="", ylabel='concentration', file_formats=['png'],
        outputCSV=None, ymin=None, ymax=None, xticking='auto',
        date_format='%Y-%m-%d', plot_type=None, envelope=False,
        black_axes=False, thresh=None, thresh_kw={},
        nb_of_minor_ticks=(2, 2), obs_style={'color': 'k', 'alpha': 0.5},
        annotation=None, min_nb_sta=1):
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
    forecastDay : int
        Forecast day for which to plot the data.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    labels : None or list of str
        List of labels for the legend.
    title : str
        Title for the plot.
    ylabel : str
        Label for the y axis.
    file_formats : list of str
        List of file extensions.
    outputCSV : str or None
        File where to save the data. The file name must contain {model}
        instead of the model name.
    ymin, ymax : None or scalar
        Limits of the axes.
    xticking : str
        Defines the method used to set x ticks. It can be 'auto' (automatic),
        'mondays' (a tick every monday) or 'daily' (a tick everyday).
    date_format : str
        String format for dates as understood by python module datetime.
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
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    thresh : None or float
        If not None, a horizontal line y = val is drawn (only if the line
        would appear inside the y axis limits).
    thresh_kw : dict
        Additional keyword arguments passed to matplotlib when drawing the
        threshold line (only used if thresh argument is not None) e.g.
        {'color': '#FFA600'}.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    min_nb_sta : int
        Minimal number of value required to compute the median or mean of
        all stations. Ingored if plot_type == None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)
    if station_list is None:
        station_list = reduce(np.intersect1d, [obj.simDF[0].columns
                                               for obj in objects])
    if 'label' not in obs_style:
        obs_style['label'] = 'observations'

    # plotting
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # first object in the list defines abscissa boundaries
    if start_end is None:
        start_end = [objects[0].startDate, objects[0].endDate]

    # find indices where we have no missing values in every object
    # complex to manage objects with =! step or =! period
    if plot_type in ['median', 'mean']:
        nan_idx = pl._find_any_missing(
            objects, labels, forecast_day=forecastDay, start_end=start_end,
            station_list=station_list,
        )

    # sim plotting
    handles = []
    for lb, c, ls, m, obj in zip(labels, colors, linestyles, markers, objects):
        simDF = evt.evaluator.Evaluator.get_sim(
            obj, forecastDay=forecastDay, start_end=start_end)[station_list]
        if plot_type in ['median', 'mean']:
            simDF[nan_idx[lb]] = np.nan
        curves = _Curves(pd.to_datetime(simDF.index), ax=ax)
        for sta, y in simDF.items():
            curves.add_curve(sta, y, {'color': c, 'linestyle': ls,
                                      'marker': m})
        if plot_type == 'median':
            curves.plot_median(style={'color': c, 'linestyle': ls,
                                      'marker': m},
                               envelope=envelope,
                               min_val=min_nb_sta)
        elif plot_type == 'mean':
            curves.plot_mean(style={'color': c, 'linestyle': ls,
                                    'marker': m},
                             envelope=envelope,
                             min_val=min_nb_sta)
        else:
            curves.plot(legend=False)
        handles.append(mlines.Line2D([], [], color=c, label=lb))
        if outputCSV is not None:
            curves.write_csv(outputCSV, model=lb)

    plt.ylabel(ylabel)
    ax.set_title(title, loc='left', x=0.05)

    # obs are plotted only for the first object in the list
    if type(objects[0]) is evt.evaluator.Evaluator:
        obsDF = objects[0].get_obs(forecastDay=forecastDay,
                                   start_end=start_end)[station_list]
        if plot_type in ['median', 'mean']:
            obsDF[nan_idx[labels[0]]] = np.nan
        curves = _Curves(pd.to_datetime(obsDF.index), ax=ax)
        for sta, y in obsDF.items():
            curves.add_curve(sta, y, obs_style)
        if plot_type == 'median':
            curves.plot_median(style=obs_style,
                               envelope=envelope,
                               min_val=min_nb_sta)
        elif plot_type == 'mean':
            curves.plot_mean(style=obs_style,
                             envelope=envelope,
                             min_val=min_nb_sta)
        else:
            curves.plot(legend=False)
        handles.append(mlines.Line2D([], [], **obs_style))
        if outputCSV is not None:
            curves.write_csv(outputCSV, model=obs_style['label'])

    # legend
    plt.figlegend(handles=handles, ncol=legend_ncol)

    # set y boundaries
    ax.set_ylim(bottom=ymin, top=ymax)

    # set xticks
    plt.xlim(left=curves.x[0], right=curves.x[-1])
    curves.set_time_xaxis(xticking, date_format)
    _set_minor_ticks(*nb_of_minor_ticks)

    # draw grid
    ax.grid(which='both', ls='-', c='#B0B0B0')

    # try to control number of yticks
    plt.locator_params(axis='y', nbins=10, min_n_ticks=10)

    # paint in black y=0 and x=0
    if black_axes is True:
        _black_axes(ax)

    # draw a hline corresponding to thresh if inside ylim
    if thresh is not None:
        if ax.axes.get_ylim()[0] < thresh < ax.axes.get_ylim()[1]:
            plt.axhline(y=thresh, **thresh_kw)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    plt.tight_layout()
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_stationScoreDensity', pl.plot_station_score_density)
def plot_stationScoreDensity(
        objects, score, forecastDay=0, labels=None,
        colors=None, linestyles=None, outputFile=None, title="",
        file_formats=['png'], availability_ratio=0.75, nb_stations=False,
        annotation=None):
    """
    Plot the probability density of a score computed per station.

    This function is based on kernel density estimation.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score : str
        Computed score.
    forecastDay : int.
        Integer corresponding to the chosen forecastDay used for plotting.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    file_formats : list of str
        List of file extensions.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute the mean scores.
    nb_stations : bool
        If True, the number of stations used to estimate the density is
        displayed in the legend.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    markers = None
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)

    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    for obj, lab, c, ls in zip(objects, labels, colors, linestyles):
        # compute scores
        df = obj.stationScores(
            [score],
            availability_ratio=availability_ratio)['D{}'.format(forecastDay)]
        # plotting
        if nb_stations is True:
            df.columns = [lab + " ({} stations)".format(
                          sum(df[score].notna()))]
        else:
            df.columns = [lab]
        try:
            df.plot.kde(ax=ax, color=c, linestyle=ls)
            ax.set_ylabel(tr.station_score_density['ylabel'][pl.LANG])
        except (ValueError, np.linalg.linalg.LinAlgError):
            print(("Warning: KDE for {} can't be computed (scores " +
                   "may contain Inf values)").format(lab))

    ax.grid(which='both', ls='-', c='#B0B0B0')
    plt.title(title)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_dataDensity', pl.plot_data_density)
def plot_dataDensity(objects, forecastDay=0, labels=None, colors=None,
                     linestyles=None, outputFile=None, title="",
                     file_formats=['png'], xmin=None, xmax=None,
                     obs_style={'color': 'k', 'alpha': 0.5},
                     annotation=None):
    """
    Plot the probability density function.

    Draw the probability density of observed (for the first object only
    and simulated data. This function is based on kernel density estimation.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    forecastDay : int.
        Integer corresponding to the chosen forecastDay used for plotting.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    title : str
        Title for the plots. It must contain {score} instead of the
        score name.
    file_formats : list of str
        List of file extensions.
    xmin, xmax : None or scalar
        Limits of the x axis.
    obs_style : dict
        Dictionary of arguments passed to pyplot.plot for observation
        curve. Default value is {'color': 'k', 'alpha': 0.5}.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define default parameters
    markers = None
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)
    if 'label' not in obs_style:
        obs_style['label'] = 'observations'

    # find indices where we have no missing value in every objects
    nan_idx = pl._find_any_missing(objects, labels, forecast_day=forecastDay)

    # plotting
    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    for obj, lab, c, ls in zip(objects, labels, colors, linestyles):
        # compute scores
        sim = obj.get_sim(forecastDay=forecastDay)
        sim[nan_idx[lab]] = np.nan
        df = pd.DataFrame(sim.values.flatten())
        # plotting
        df.columns = [lab]
        try:
            df.plot.kde(ax=ax, color=c, linestyle=ls)
            ax.set_ylabel(tr.station_score_density['ylabel'][pl.LANG])
        except (ValueError, np.linalg.linalg.LinAlgError):
            print(("Warning: KDE for {} can't be computed (scores " +
                   "may contain Inf values)").format(lab))

    # obs are plotted only for the first object in the list
    obs = objects[0].get_obs(forecastDay=forecastDay)
    obs = obs.where(~nan_idx[labels[0]], np.nan)
    df = pd.DataFrame(obs.values.flatten())
    df.columns = [obs_style['label']]
    try:
        df.plot.kde(ax=ax, **obs_style)
        ax.set_ylabel(tr.station_score_density['ylabel'][pl.LANG])
    except (ValueError, np.linalg.linalg.LinAlgError):
        print("Warning: KDE for observations can't be computed " +
              "(time series may contain Inf values).")

    ax.grid(which='both', ls='-', c='#B0B0B0')
    plt.title(title)
    ax.set_xlim(left=xmin, right=xmax)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')
    # save figure
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_barScores', pl.plot_bar_scores)
def plot_barScores(
        objects, score_list, forecastDay=0, averaging='mean', title="",
        labels=None, colors=None, subregions=None, xticksLabels=None,
        file_formats=['png'], outputFile=None, outputCSV=None,
        availability_ratio=.75, bar_kwargs={}, annotation=None,
        ref_line=None):
    """
    Barplot for scores.

    Draw one barplot per score, with one bar per object. If there are
    subregions, one set of bars per region will be drawn in each barplot.

    Scores are first computed for each measurement sites and then averaged.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score_list : list of str
        Computed scores.
    forecastDay : int
        Integer corresponding to the chosen forecastDay used for plotting.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Bar colors corresponding to each objects.
    subregions : None or list of list of str
        One set of bars per sub-region will be drawn. The sub-regions must be
        given like [['FR'], ['ES', 'FR'], 'all', ...], where 'FR', 'ES', ...
        are the first letters of the station codes you want to keep
        and 'all' means that all stations are kept.
    xticksLabels : None or list of str
        List of labels for the xticks. These labels corresponds to sub-region
        define *subregions* argument. Labels can contain '{nbStations}' that
        will be replaced by the corresponding number of stations used to
        compute the score (warning: if several objects are displayed, the
        number of stations corresponds to the first one).
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    outputCSV : str or None
        File where to save the data.
    file_formats : list of str
        List of file extensions.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        term to compute the mean scores.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    ref_line : dict of args or None
        Plot an horizontal line whose arguments are passed to
        matplotlib.pyplot.hline.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects, False)
    pl._check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, None, None)
    if subregions is None:
        subregions = ["all"]
    if xticksLabels is None:
        xticksLabels = np.full(len(subregions), '')

    if len(subregions) != len(xticksLabels):
        raise evt.EvaltoolsError("Warning: 'subregions' and 'xticksLabels'" +
                                 " must have same length.")

    # stats computing
    stats = [obj.stationScores(score_list=score_list,
                               availability_ratio=availability_ratio)[
                                    'D{}'.format(forecastDay)]
             for obj in objects]

    # defining averaging function
    if averaging == 'mean':
        average = pd.DataFrame.mean
    elif averaging == 'median':
        average = pd.DataFrame.median
    else:
        raise evt.EvaltoolsError("averaging argument must be either equal " +
                                 "to 'mean' or 'median'.")

    # plotting
    plt.close('all')
    figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(
        figsize=(figsize[0]*len(score_list), figsize[1]),
        constrained_layout=False,
    )

    for sc in score_list:
        ax = fig.add_subplot(1, len(score_list), score_list.index(sc)+1)
        df = pd.DataFrame()
        nbStations = []
        for i, region in enumerate(subregions):
            if region == 'all':
                sub_df = pd.DataFrame({lab: y[sc]
                                       for lab, y in zip(labels, stats)})
            else:
                kept_stations = [sta for sta in objects[0].stations.index
                                 if any([re.match(cn, sta) for cn in region])]
                if len(kept_stations) == 0:
                    print("Warning: subregion {subreg} is empty !!!".format(
                                subreg=region))
                sub_df = pd.DataFrame({lab: y[sc].loc[kept_stations]
                                       for lab, y in zip(labels, stats)})
            nbStations.append(np.sum(sub_df[labels[0]].notnull()))
            df[i] = average(sub_df)

        # set tick labels
        df.columns = [lab.format(nbStations=nbStations[i])
                      for i, lab in enumerate(xticksLabels)]
        ax.tick_params('y', labelsize='small')

        # ylabel
        ax.set_ylabel(sc)

        # to respect variable order
        df = df.loc[labels]

        # save in CSV format
        if outputCSV is not None:
            df.T.to_csv(path_or_buf=outputCSV.format(score=sc), sep=" ",
                        na_rep="nan")

        # xticks format
        plt.xticks(rotation=45, ha='right')

        # barplot
        df.T.plot.bar(ax=ax, color=colors, legend=False, **bar_kwargs)

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

    # figure legend
    plt.figlegend(handles=[mlines.Line2D([], [], color=c, label=l, linewidth=3)
                           for c, l in zip(colors, labels)],
                  ncol=legend_ncol)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # title
    fig.suptitle(title)

    # save figure
    _save_figure(outputFile, file_formats, bbox_inches='tight')

    return plt.gcf(), plt.gca()


@deprecate_func('plot_barExceedances', pl.plot_bar_exceedances)
def plot_barExceedances(
        obj, threshold, data="obs", start_end=None, forecastDay=0,
        labels=None, title="", ylabel=None, outputFile=None,
        file_formats=['png'], outputCSV=None, subregions=None, ymin_max=None,
        dpi=100, xticking='daily', date_format='%Y-%m-%d', bar_kwargs={},
        annotation=None):
    """
    Barplot for threshold exceedances.

    Draws a barplot of threshold exceedances for the period defined by
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
    forecastDay : int.
        Integer corresponding to the chosen forecastDay used for plotting.
    labels : None or list of str
        List of labels for the legend.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    title : str
        Title for the plot. May contain {total} to print total number of
        obs/sim that exceed threshold.
    ylabel : str
        Ordinate axis label.
    file_formats : list of str
        List of file extensions.
    subregions : None or list of list of str
        One set of bars per sub-region will be drawn. The sub-regions must be
        given like [['FR'], ['ES', 'FR'], 'all', ...], where 'FR', 'ES', ...
        are the first letters of the station codes you want to keep
        and 'all' means that all stations are kept.
    ymin_max : None or list of two scalars
        Limits of the y axis. The second scalar is not included in the y axis !
    dpi : integer
        Figure resolution (may solve graphical issues when plotting too many
        bars).
    xticking : str
        Defines the method used to set x ticks. It can be either 'daily',
        'mondays' or 'bimonthly'.
    date_format : str
        String format for dates as understood by python module datetime.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # get some useful variables
    if start_end is None:
        start_end = [obj.startDate, obj.endDate]
    if ylabel is None:
        ylabel = tr.bar_exceedances['ylabel'][pl.LANG]

    plt.close('all')
    fig = plt.figure(dpi=dpi, constrained_layout=True)
    ax = plt.gca()

    # subsetting: one object per region
    if subregions is None:
        objects = [obj]
    else:
        objects = [obj if region == 'all'
                   else obj.selectCountries(region, False)
                   for region in subregions]

    days_list = evt.dataset.timeRange(
        start_end[0],
        start_end[1],
        seriesType='daily').date
    bars_heights = pd.DataFrame()
    for i, objct in enumerate(objects):
        for day in days_list:
            obs = objct.get_obs(forecastDay, (day, day))
            sim = objct.get_sim(forecastDay, (day, day))
            res = evt.scores.contingency_table(obs.values.flatten(),
                                               sim.values.flatten(),
                                               thr=threshold)
            if data == "sim":
                bars_heights.loc[day + dt.timedelta(days=forecastDay),
                                 i] = res[2, 0]
            else:
                bars_heights.loc[day + dt.timedelta(days=forecastDay),
                                 i] = res[0, 2]

    if labels is not None:
        bars_heights.columns = labels
    bars_heights.plot.bar(ax=ax,
                          stacked=True,
                          legend=(labels is not None),
                          **bar_kwargs)

    # save in CSV format
    if outputCSV is not None:
        bars_heights.to_csv(path_or_buf=outputCSV, sep=" ", na_rep="nan")

    # Customisation
    plt.ylabel(ylabel)
    tot = evt.scores.contingency_table(
        obj.get_obs(forecastDay).values.flatten(),
        obj.get_sim(forecastDay).values.flatten(),
        thr=threshold)
    total = tot[2, 0] if data == "sim" else tot[0, 2]
    plt.title(title.format(total=int(total)), loc='center')
    if ymin_max is not None:
        ax.set_ylim(bottom=ymin_max[0], top=ymin_max[1])

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # set xticks
    if xticking == 'daily':
        xtick_labels = [d.strftime(date_format) for d in bars_heights.index]
        xtick_pos = list(range(len(days_list)))
    elif xticking == 'mondays':
        xtick_labels = [d.strftime(date_format) for d in bars_heights.index
                        if d.weekday() == 0]
        xtick_pos = [i for i, d in enumerate(bars_heights.index)
                     if d.weekday() == 0]
    elif xticking == 'bimonthly':
        xtick_labels = [d.strftime(date_format) for d in bars_heights.index
                        if d.day in [0, 15]]
        xtick_pos = [i for i, d in enumerate(bars_heights.index)
                     if d.day in [0, 15]]
    else:
        raise evt.EvaltoolsError("xticking argument must be either " +
                                 "'daily', 'mondays' or 'bimonthly'")
    plt.xticks(xtick_pos, xtick_labels)

    # save figure
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_lineExceedances', pl.plot_line_exceedances)
def plot_lineExceedances(
        objects, threshold, start_end=None, forecastDay=0,
        labels=None, colors=None, linestyles=None, markers=None,
        outputFile=None, title="", ylabel=None,
        file_formats=['png'], dpi=100, xticking='daily',
        date_format=None, ymin=None, ymax=None,
        obs_style={'color': 'k', 'alpha': 0.5}, outputCSV=None,
        black_axes=False, nb_of_minor_ticks=(1, 2), annotation=None):
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
    forecastDay : int.
        Integer corresponding to the chosen forecastDay used for plotting.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Line colors corresponding to each objects.
    linestyles : None or list of str
        Line styles corresponding to each objects.
    markers : None or list of str
        Line markers corresponding to each objects.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    title : str
        Title for the plot.
    ylabel : str
        Ordinate axis label.
    file_formats : list of str
        List of file extensions.
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
    annotation : str or None
        Additional information to write in figure's upper left corner.
    outputCSV : str or None
        File where to save the data.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the station list
    pl._check_station_list(objects)

    # define default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, linestyles, markers)
    if 'label' not in obs_style:
        obs_style['label'] = 'observations'
    if ylabel is None:
        ylabel = tr.line_exceedances['ylabel'][pl.LANG]

    # get period
    if start_end is None:
        pl._check_period(objects)
        start_end = [objects[0].startDate, objects[0].endDate]
    days_list = evt.dataset.timeRange(
        start_end[0],
        start_end[1],
        seriesType='daily').date

    # compute exceedances
    df = pd.DataFrame(
        columns=[obj.model for obj in objects] + [obs_style['label']])
    for obj in objects:
        for day in days_list:
            obs = obj.get_obs(forecastDay, (day, day))
            sim = obj.get_sim(forecastDay, (day, day))
            res = evt.scores.contingency_table(obs.values.flatten(),
                                               sim.values.flatten(),
                                               thr=threshold)
            df.loc[day + dt.timedelta(days=forecastDay), obj.model] = res[2, 0]
            if obj is objects[0]:
                df.loc[day + dt.timedelta(days=forecastDay),
                       obs_style['label']] = res[0, 2]

    # plotting
    plt.close('all')
    curves = _Curves(df.index, constrained_layout=False)
    for o, l, c, ls, m in zip(objects, labels, colors, linestyles, markers):
        if df[o.model].isna().all():
            print("*** all nan values for {label}".format(label=l))
        else:
            curves.add_curve(
                l,
                df[o.model],
                style={'label': l, 'color': c, 'linestyle': ls, 'marker': m},
            )
    curves.add_curve(obs_style['label'], df[obs_style['label']],
                     style=obs_style)
    curves.plot(legend=True, legend_type='ax', black_axes=black_axes)
    plt.ylabel(ylabel)
    plt.title(title, loc='center')

    # annotation
    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    curves.ax.grid(which='both', ls='-', c='#B0B0B0')
    curves.ax.set_ylim(bottom=ymin, top=ymax)

    # set xticks
    curves.set_time_xaxis(xticking, date_format)
    _set_minor_ticks(*nb_of_minor_ticks)

    # save figure
    plt.tight_layout()
    _save_figure(outputFile, file_formats)

    # save data
    if outputCSV is not None:
        curves.write_csv(outputCSV)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_barContingencyTable', pl.plot_bar_contingency_table)
def plot_barContingencyTable(
        objects, threshold, forecastDay=0, start_end=None, title="",
        labels=None, ymin_max=None, outputFile=None, file_formats=['png'],
        outputCSV=None, bar_kwargs={}, annotation=None):
    """
    Barplot for contingency table.

    For each object, draws a bar for good detections, false
    alarms, and missed alarms.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    threshold : scalar
        Threshold value.
    forecastDay : int
        Integer corresponding to the chosen forecastDay used for plotting.
    start_end : None or list of two datetime.date objects
        Boundary dates for the studied period.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels coresponding to each object.
    ymin_max : None or list of two scalars
        Limits of the y axis.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    outputCSV : str or None
        File where to save the data.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    annotation : str or None
        Additional information to write in figure's upper left corner.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects)
    pl._check_station_list(objects)

    # define objects labels in case of None
    if labels is None:
        labels = []
        for obj in objects:
            labels.append(obj.model)

    # stats computing
    df = pd.DataFrame(index=tr.bar_contingency_table['index'][pl.LANG])
    for obj, lab in zip(objects, labels):
        obs = obj.get_obs(forecastDay=forecastDay,
                          start_end=start_end)
        sim = obj.get_sim(forecastDay=forecastDay,
                          start_end=start_end)
        tab = evt.scores.contingency_table(obs.values.flatten(),
                                           sim.values.flatten(),
                                           thr=threshold)
        df[lab] = [tab[0, 0], tab[1, 0], tab[0, 1]]

    # plotting
    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    ax = plt.gca()
    ax.tick_params('y', labelsize='small')

    # barplot
    df.T.plot.bar(ax=ax, legend=True, **bar_kwargs)

    # save in CSV format
    if outputCSV is not None:
        df.T.to_csv(path_or_buf=outputCSV, sep=" ", na_rep="nan")

    # layout cutomisation
    if ymin_max is not None:
        ax.set_ylim(bottom=ymin_max[0], top=ymin_max[1])
    plt.title(title, loc='center')

    # annotation
    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('plot_barScoresConc', pl.plot_bar_scores_conc)
def plot_barScoresConc(
        objects, score_list, conc_range, forecastDay=0, averaging='mean',
        title="", labels=None, colors=None, xticksLabels=None,
        outputFile=None, file_formats=['png'], outputCSV=None,
        min_nb_val=10, based_on='obs', bar_kwargs={},
        annotation=None, nb_vals=True, subplots_adjust=None):
    """
    Barplot for scores per concentration class.

    Data is grouped depending on the desired concentration classes, then
    scores are computed for each site and averaged.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    score_list : list of str
        Computed scores.
    conc_range : list of scalars
        List used to determine concentration intervals on which to compute
        scores. Must contain at least two values. e.g [25, 45, 80] determines
        scores with concentrations between 25 and 45, and between 45 and 80.
    forecastDay : int
        Integer corresponding to the chosen forecastDay used for plotting.
    averaging : str
        Type of score averaging choosen among 'mean' or 'median'.
    title : str
        Title for the figure.
    labels : None or list of str
        List of labels for the legend.
    colors : None or list of str
        Bar colors corresponding to each objects.
    xticksLabels : None or list of str
        List of labels for the xticks.
    outputFile : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    outputCSV : str or None
        File where to save the data. The file name must contain {score}
        instead of the score name.
    min_nb_val : int
        Minimal number of (obs, sim) couple required for a score to be
        computed.
    based_on : str
        If 'sim', concentrations are determined from simulation data. Else
        ('obs') they are determined with observations.
    bar_kwargs : dict
        Additional keyword arguments passed to pandas.DataFrame.plot.bar.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    nb_vals : boolean
        Whether the number of computed values for each bar must be displayed
        or not.
    subplots_adjust : dict
        Keyword arguments passed to matplotlib.pyplot.subplots_adjust.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects, False)
    pl._check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, None, None)

    # stats computing
    stats = []
    tot = []
    for i in range(len(conc_range)-1):
        stats.append([])
        tot.append([])
        stats[i], tot[i] = zip(
            *[obj.conc_scores(
                        score_list=score_list,
                        conc_range=[conc_range[i], conc_range[i+1]],
                        min_nb_val=min_nb_val,
                        based_on=based_on,
                        forecastDay=forecastDay,
                    )
              for obj in objects]
        )

    if nb_vals is True:
        total = [tot[j][0] for j in range(len(tot))]
        total = np.round(np.true_divide(total, sum(total))*100, decimals=1)
        total = [str(t)+'%' for t in total]
        tot = str(sum(tot[:][0]))

    if xticksLabels is None:
        xticksLabels = [str(c1)+' <= c < '+str(c2)
                        for c1, c2 in zip(conc_range[:-1],
                                          conc_range[1:])]
        # xticksLabels[-1] = str(conc_range[-2])+' <= c'

    # defining averaging function
    if averaging == 'mean':
        average = pd.DataFrame.mean
    elif averaging == 'median':
        average = pd.DataFrame.median
    else:
        raise evt.EvaltoolsError("averaging argument must be either equal " +
                                 "to 'mean' or 'median'.")

    # plotting
    plt.close('all')
    figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(figsize[0]*len(score_list), figsize[1]))

    for sc in score_list:
        ax = fig.add_subplot(1, len(score_list), score_list.index(sc)+1)
        df = pd.DataFrame()
        for i in range(len(conc_range)-1):
            sub_df = pd.DataFrame({lab: y[sc]
                                   for lab, y in zip(labels, stats[i])})
            df[i] = average(sub_df)

        # set tick labels
        df.columns = [xticksLabels[i] for i in range(len(conc_range)-1)]
        ax.tick_params('y', labelsize='small')

        # ylabel
        ax.set_ylabel(sc)

        # to respect objects order
        df = df.loc[labels]

        # barplot
        axes = df.T.plot.bar(ax=ax, color=colors, legend=False, **bar_kwargs)

        if outputCSV is not None:
            df.T.to_csv(outputCSV.format(score=sc), sep=' ', na_rep='nan')

        # print tot values
        if nb_vals is True:
            pos = [1./len(total)/2.]
            for x in range(len(total)-1):
                pos.append(pos[-1]+1./len(total))
            for x in range(len(total)):
                ax.text(pos[x], 1, str(total[x]), fontsize=8, color="#7A7A7A",
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        transform=ax.transAxes)
            plt.annotate(
                tr.bar_scores_conc['annotation'][pl.LANG].format(tot),
                xy=(0, 1),
                xytext=(0, 17),
                xycoords='axes fraction',
                textcoords='offset points',
                va='top',
                fontsize=8,
                fontstyle='italic',
                color='#7A7A7A',
            )

    # figure legend
    handles = [mlines.Line2D([], [], color=c, label=l, linewidth=3)
               for c, l in zip(colors, labels)]
    plt.figlegend(
        handles=handles,
        ncol=legend_ncol,
    )
    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')
    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)

    # title
    fig.suptitle(title)

    # save figure
    # plt.tight_layout()
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('roc_curve', pl.plot_roc_curve)
def roc_curve(objects, thresholds, forecastDay=0,
              labels=None, colors=None, markers=None,
              outputFile=None, file_formats=['png'],
              suptitle="", title="ROC diagram", xylabels={}, start_end=None,
              annotation=None):
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
    forecastDay : int
        Forecast day for which to use data to compute the ROC diagram.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Marker colors corresponding to each objects.
    markers : None or list of str
        Marker shapes corresponding to each objects.
    outputFile : str
        File where to save the plot (without extension).
    file_formats : list of str
        List of file extensions.
    suptitle : str
        Chart suptitle.
    title : str
        Chart title.
    xylabels : dict
        Dict of str with keys equal to 'x' (abscissa) and/or 'y' (ordinate)
        labels. Default labels are taken for keys absent from dictionary.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects, False)
    pl._check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, None, markers, 'x')
    xlabel = xylabels.get('x', 'POFD')
    ylabel = xylabels.get('y', 'POD')

    # begin plotting
    plt.close('all')
    fig = plt.figure(dpi=125, constrained_layout=True)
    ax = plt.gca()

    # get performance scores for each object
    pod = []
    pofd = []
    for obj in objects:
        prob_detect = [0.]
        prob_false_detect = [0.]
        for thr in reversed(np.sort(thresholds)):
            perf = evt.scores.contingency_table(
                obj.get_obs(forecastDay, start_end=start_end).values.flatten(),
                obj.get_sim(forecastDay, start_end=start_end).values.flatten(),
                thr=thr,
            )
            prob_detect.append(perf[0, 0]/perf[0, 2] if perf[0, 2] != 0
                               else np.nan)
            prob_false_detect.append(perf[1, 0]/perf[1, 2] if perf[1, 2] != 0
                                     else np.nan)
        prob_detect.append(1.)
        prob_false_detect.append(1.)
        pod.append(prob_detect)
        pofd.append(prob_false_detect)

    # reference x=y line
    ax.plot([0, 1], [0, 1], 'ko-', label='No skill (SSr=0)')

    # plot points and lines for each object
    for i, obj in enumerate(objects):
        area = np.trapz(y=pod[i], x=pofd[i]) / 0.5 - 1.0
        area = str(round(area, 2))
        ax.plot(pofd[i], pod[i],
                marker=markers[i], color=colors[i],
                label=labels[i] + " (SSr={a})".format(a=area))

    # Titles, labels and legend
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    ax.legend(loc='lower right')

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    _save_figure(outputFile, file_formats)

    return plt.gcf(), plt.gca()


@deprecate_func('performance_diagram', pl.plot_performance_diagram)
def performance_diagram(
        objects, threshold, forecastDay=0, labels=None, colors=None,
        markers=None, outputFile=None, file_formats=['png'],
        suptitle="", title="Performance Diagram", xylabels={},
        annotation=None, start_end=None):
    """
    Draw a performance diagram.

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
    forecastDay : int
        Forecast day for which to use data to compute the diagram.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Marker colors corresponding to each objects.
    markers : None or list of str
        Marker shapes corresponding to each objects.
    outputFile : str
        File where to save the plot (without extension).
    file_formats : list of str
        List of file extensions.
    suptitle : str
        Diagram suptitle.
    title : str
        Diagram title.
    xylabels : dict
        Dict of str with keys equal to 'x' (abscissa), 'y' (ordinate), 'b'
        (colorbar) and/or 'f' (dotted lines) labels. Default labels are taken
        for keys absent from dictionary.
    annotation : str or None
        Additional information to write in figure's upper left corner.
    start_end : None or list of two datetime.date objects
        Boundary dates used to select only data for a sub-period.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    # check if objects are defined on the same period and same stations
    pl._check_period(objects, False)
    pl._check_station_list(objects)

    # get default parameters
    labels, colors, linestyles, markers, legend_ncol, seriesType = \
        pl._default_params(objects, labels, colors, None, markers, 'o')
    xlabel = xylabels.get('x', tr.performance_diagram['xlabel'][pl.LANG])
    ylabel = xylabels.get('y', tr.performance_diagram['ylabel'][pl.LANG])
    csi_label = xylabels.get('b', tr.performance_diagram['csi_label'][pl.LANG])
    freq_label = xylabels.get(
        'f', tr.performance_diagram['freq_label'][pl.LANG],
    )

    # begin plotting
    plt.close('all')
    fig = plt.figure(dpi=125)
    ax = plt.gca()

    ax.grid(False, which='both')

    # get performance scores for each object
    prob_detect = []
    success_ratio = []
    for obj in objects:
        perf = evt.scores.contingency_table(
            obj.get_obs(forecastDay, start_end=start_end).values.flatten(),
            obj.get_sim(forecastDay, start_end=start_end).values.flatten(),
            thr=threshold,
        )
        prob_detect.append(
            perf[0, 0]/perf[0, 2] if perf[0, 2] != 0 else np.nan)
        success_ratio.append(
            perf[0, 0]/perf[2, 0] if perf[2, 0] != 0 else np.nan)

    # grid for background colors and lines
    grid_ticks = np.arange(0, 1.001, 0.001)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)

    # critical success index background
    csi = np.zeros_like(sr_g)
    mask = (sr_g*pod_g == 0)
    csi[~mask] = 1.0/(1.0/sr_g[~mask] + 1.0/pod_g[~mask] - 1.0)
    csi_contour = plt.contourf(sr_g, pod_g, csi, np.arange(0., 1.1, 0.05),
                               extend="max", cmap="RdYlGn", alpha=0.5)
    cbar = plt.colorbar(csi_contour)

    # frequency bias lines
    bias = np.full_like(sr_g, np.nan)
    mask = (sr_g == 0)
    bias[~mask] = pod_g[~mask]/sr_g[~mask]
    freq_lines = plt.contour(sr_g, pod_g, bias,
                             [0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2, 4],
                             colors="k", linestyles="dashed")
    plt.clabel(
        freq_lines,
        fmt="%1.1f",
        manual=[(.8, .09), (.85, .21), (.9, .45), (.92, .7), (.94, .94),
                (.75, .95), (.6, .92), (.44, .89), (.21, .83)],
    )

    # plot each point
    for i, obj in enumerate(objects):
        ax.scatter(success_ratio[i], prob_detect[i],
                   marker=markers[i], c=colors[i],
                   facecolors='none', label=labels[i],
                   zorder=10)

    # Titles, labels and legend
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    cbar.set_label(csi_label, fontsize=14)
    freq_lines.collections[4].set_label(freq_label)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    legend_params = dict(bbox_to_anchor=(1.22, 1), loc='upper left',
                         fontsize=12, framealpha=1, frameon=True)
    plt.legend(**legend_params)

    if annotation is not None:
        plt.annotate(annotation,
                     xy=(0, 1), xycoords='figure fraction', va='top',
                     fontsize='small', fontstyle='italic', color='#7A7A7A')

    # save figure
    _save_figure(outputFile, file_formats, bbox_inches='tight')

    return plt.gcf(), plt.gca()


plot_comparisonScatterPlot = deprecate(
    'plot_comparisonScatterPlot',
    pl.plot_comparison_scatter_plot,
)
plot_significantDifferences = deprecate(
    'plot_significantDifferences',
    pl.plot_significant_differences,
)
plot_diurnalCycle = deprecate(
    'plot_diurnalCycle',
    pl.plot_diurnal_cycle,
)
plot_valuesScatterPlot = deprecate(
    'plot_valuesScatterPlot',
    pl.plot_values_scatter_plot,
)
