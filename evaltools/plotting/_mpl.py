# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Plotting tools using matplotlib."""
from itertools import repeat
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
import matplotlib.markers as mmarkers

import evaltools as evt
import evaltools.plotting as pl
import evaltools.translation as tr

pd.plotting.register_matplotlib_converters()


def save_figure(
        output_file, file_formats, bbox_inches=None, tight_layout=False):
    """
    Save the figure in different formats or show it if output_file is None.

    Parameters
    ----------
    output_file : None or str
        Path of the file where to save the plots (without extension).
    file_formats : list of str
        List of file extensions among 'svgz', 'ps', 'emf', 'rgba', 'raw',
        'pdf', 'svg', 'eps' or 'png'.

    """
    if tight_layout:
        plt.tight_layout()

    if output_file is None:
        if pl.USER_INTERFACE_WINDOW:
            plt.show()
    else:
        for ext in file_formats:
            plt.savefig(
                (output_file+".{ext}").format(ext=ext),
                bbox_inches=bbox_inches,
            )
        plt.close()


def draw_black_axes(ax):
    """Paint in black y=0 and x=0."""
    if ax.axes.get_xlim()[0] < 0 and ax.axes.get_ylim()[1] > 0:
        plt.axvline(x=0, color='k')
    if ax.axes.get_ylim()[0] < 0 and ax.axes.get_ylim()[1] > 0:
        plt.axhline(y=0, color='k')


def hline(ax, val, **kwargs):
    """Draw a horizontal line if inside y limits."""
    if ax.axes.get_ylim()[0] < val < ax.axes.get_ylim()[1]:
        plt.axhline(y=val, **kwargs)


def mscatter(x, y, ax=None, m=None, **kw):
    """
    Make a scatter plot with different point markers.

    Pyplot scatter plot function with the possibility to use a different
    marker for each point.

    Parameters
    ----------
    x, y : 1D array
        The data positions.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot.
    m : 1D array of str
        The markers for every points.
    kw : dict
        Keyword arguments passed to matplotlib.pyplot.scatter.

    """
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
                    "_mscatter function can not manage nan values in c "
                    "argument."
                )
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


def is_outlier(points, thresh=3.5):
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


def set_minor_ticks(ax, nb_x, nb_y):
    """
    Define number of minor ticks between each major one.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axis to set up.
    nb_x, nb_y : int
        Number of minor ticks for x and y axes.

    """
    minor_locator = AutoMinorLocator(n=nb_x)
    ax.xaxis.set_minor_locator(minor_locator)
    minor_locator = AutoMinorLocator(n=nb_y)
    ax.yaxis.set_minor_locator(minor_locator)


def set_time_xaxis(fig, ax, xticking, date_format=None):
    """
    Set pyplot x axis as a datetime axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to set up.
    ax : matplotlib.axes._axes.Axes
        Axis to set up.
    xticking : str
        Defines the method used to set x ticks. It can be 'auto' (automatic),
        'mondays' (a tick every monday) or 'daily' (a tick everyday).
    date_format : str
        String format for dates as understood by python module datetime.

    """
    if date_format is not None:
        fmt = mdates.DateFormatter(date_format)
        ax.xaxis.set_major_formatter(fmt)
    if xticking != 'auto':
        if xticking == 'daily':
            loc = mdates.WeekdayLocator(byweekday=range(7))
        elif xticking == 'mondays':
            loc = mdates.WeekdayLocator(byweekday=0)
        elif xticking == 'bimonthly':
            loc = mdates.WeekdayLocator(byweekday=1, interval=2)
        else:
            raise evt.EvaltoolsError(
                "xticking argument must be either 'daily', 'mondays' "
                "'bimonthly' or 'auto'"
            )
        ax.xaxis.set_major_locator(loc)
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tick_params(axis='x', which='major', labelsize='small')


def mpl_plot_line(
        data, plot_with_pandas=True, plot_kw={}, outlier_thresh=None,
        legend=True, legend_type='fig', legend_kw={},
        fig=None, ax=None):
    """
    Draw a Line2D plot of a DataFrame columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    plot_with_pandas : bool
        If True, use pandas.DataFrame.plot method, else, use several calls to
        matplotlib.pyplot.plot.
    plot_kw : dict
        Keyword arguments passed to pandas.DataFrame.plot.
    outlier_thresh : scalar or None
        If not None, it correspond to the threshold used in
        _is_outlier to determine if a model is an
        outlier. If outliers are detected, y boundaries do not take
        them into account.
    legend : bool
        If True, the legend is plotted.
    legend_type : str
        Set to 'fig' for figure legend or to 'ax' for axis legend.
    legend_kw : dict
        Keyword arguments passed to pyplot.legend or plt.figlegend.
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
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    if plot_with_pandas:
        data.plot(ax=ax, legend=False, **plot_kw)
    else:
        colors = plot_kw.pop('color', repeat(None))
        styles = plot_kw.pop('linestyle', repeat(None))
        markers = plot_kw.pop('marker', repeat(None))
        for col, c, ls, m in zip(data, colors, styles, markers):
            ax.plot(
                data.index, data[col],
                label=col, color=c, ls=ls, marker=m,
                **plot_kw,
            )

    if legend:
        if legend_type == 'fig':
            plt.figlegend(**legend_kw)
        elif legend_type == 'ax':
            plt.legend(**legend_kw)

    if outlier_thresh:
        points = data.T
        outliers = is_outlier(points, outlier_thresh)
        if outliers.any() and not outliers.all():
            # find min/max
            ymin = np.nanmin(points.loc[~outliers, :])
            ymax = np.nanmax(points.loc[~outliers, :])
            margin = (ymax - ymin)*.05
            ax.set_ylim(bottom=ymin-margin, top=ymax+margin)

    return fig, ax


def mpl_plot_box(
        data, colors, plot_kw={}, fig=None, ax=None):
    """
    Draw a boxplot of a DataFrame's columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    colors : None or list of str
        Box colors corresponding to each label.
    plot_kw : dict
        Keyword arguments passed to pandas.DataFrame.plot.box.
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
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    bplot = data.plot.box(
        ax=ax,
        return_type='dict',
        patch_artist=True,
        **plot_kw,
    )
    for artist, color in zip(bplot['boxes'], colors):
        artist.set_facecolor(color)
        artist.set_edgecolor('k')
    for artist in bplot['whiskers']:
        artist.set_color('k')
    for artist, color in zip(bplot['medians'], colors):
        artist.set_color('k' if color != 'k' else 'w')

    return fig, ax


def mpl_plot_bar(
        data, plot_kw={}, fig=None, ax=None):
    """
    Draw a barplot of a DataFrame's columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.
    plot_kw : dict
        Keyword arguments passed to pandas.DataFrame.plot.box.
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
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    data.plot.bar(
        ax=ax,
        # return_type='dict',
        # patch_artist=True,
        **plot_kw,
    )

    return fig, ax


def set_axis_elements(
        ax, xticks_kw=None, nb_of_minor_ticks=None,
        xlabel_kw=None, ylabel_kw=None, title_kw=None,
        annotate_kw=None, black_axes=False,
        xmin=None, xmax=None, ymin=None, ymax=None,
        **kwargs):
    """
    Set some components of a maplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axis to modify.
    xticks_kw : dict
        Keyword arguments passed to pyplot.xticks.
    nb_of_minor_ticks : tuple of int
        Number of minor ticks for x and y axes.
    xlabel_kw : dict
        Keyword arguments passed to matplotlib.axes.Axes.set_xlabel.
    ylabel_kw : dict
        Keyword arguments passed to matplotlib.axes.Axes.set_ylabel.
    title_kw : dict
        Keyword arguments passed to matplotlib.axes.Axes.set_title.
    annotate_kw : dict
        Keyword arguments passed to matplotlib.axes.Axes.annotate.
    black_axes : bool
        If true, y=0 and x=0 lines are painted in black.
    xmin, xmax : None or scalar
        Limits of the x axis.
    ymin, ymax : None or scalar
        Limits of the y axis.
    kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.setp.

    """
    plt.sca(ax)

    if xticks_kw:
        plt.xticks(**xticks_kw)
    if nb_of_minor_ticks:
        set_minor_ticks(ax, *nb_of_minor_ticks)

    if ylabel_kw:
        ax.set_ylabel(**ylabel_kw)
    if xlabel_kw:
        ax.set_xlabel(**xlabel_kw)

    if title_kw:
        ax.set_title(**title_kw)

    if annotate_kw:
        ax.annotate(**annotate_kw)

    if black_axes:
        draw_black_axes(ax)

    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)

    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    plt.setp(ax, **kwargs)


def set_cartopy(
        fig=None,
        ax=None,
        projection=None,
        grid_resolution=None,
        ne_features=None,
        rivers=False,
        land_color='none',
        sea_color='none',
        background_color=None,
        bnd_resolution='50m',
        boundaries_above=False,
        sea_mask=False,
        land_mask=False,
        bbox=None):
    """
    Set Cartopy layout.

    Parameters
    ----------
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or cartopy.mpl.geoaxes.GeoAxes
        Axis to use for the plot. If None, a new axis is created.
    projection : cartopy.crs projection
        Projection to be used for plot. If None, PlateCarree projection is
        used.
    grid_resolution : couple of scalars
        Couple of scalars corresponding to meridians and parallels spacing
        in degrees.
    ne_features : list of dicts
        Each dictionary contains arguments to instanciate a
        cartopy.feature.NaturalEarthFeature(...).
        E.g. [dict(category='cultural', name='admin_1_states_provinces',
        facecolor='none', linestyle=':'),] will add
        states/departments/provinces.
    rivers : bool
        If true, rivers are displayed in the map.
    land_color, sea_color : str
        Land/sea colors.
    background_color : str
        The background color.
    bnd_resolution : str
        Resolution of boundaries ('110m', '50m' or '10m').
    boundaries_above : bool
        If True, boundaries and coast lines are drawn above score data.
    sea_mask : bool
        If True, scores ought to be drawn over sea are masked.
    land_mask : bool
        If True, scores ought to be drawn over land are masked.
    bbox : Tuple of floats
        Tuple of floats representing the required extent (x0, x1, y0, y1).

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
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER

    if projection is None:
        if bbox and bbox[1] > 180:
            # to be able to use a bbox with min_lat < 180 and max_lat > 180
            projection = ccrs.PlateCarree(central_longitude=bbox[0])
        else:
            projection = ccrs.PlateCarree()

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1, projection=projection)

    if background_color:
        ax.background_patch.set_facecolor(background_color)

    # map features
    if ne_features is None:
        ne_features = []
        if rivers:
            ne_features.append(
                dict(
                    category='physical',
                    name='lakes',
                    scale=bnd_resolution,
                    edgecolor='face',
                    facecolor=np.array((152, 183, 226)) / 256.
                )
            )
            ne_features.append(
                dict(
                    category='physical',
                    name='rivers_lake_centerlines',
                    scale=bnd_resolution,
                    edgecolor=np.array((152, 183, 226)) / 256.,
                    facecolor='none',
                    alpha=0.5,
                )
            )
        # ne_features.append(
        #     {
        #         'category': 'cultural',
        #         'name': 'admin_0_countries',
        #         'facecolor': 'none',
        #         'edgecolor': 'k',
        #         'scale': bnd_resolution,
        #     },
        # )

        ne_features.append(
            dict(
                category='physical',
                name='land',
                scale=bnd_resolution,
                edgecolor='none',
                facecolor=land_color,
            )
        )
        ne_features.append(
            dict(
                category='physical',
                name='ocean',
                scale=bnd_resolution,
                edgecolor='none',
                facecolor=sea_color,
            )
        )
        ne_features.append(
            dict(
                category='physical',
                name='land',
                scale=bnd_resolution,
                edgecolor='k',
                facecolor='none',
                zorder=2 + 10*boundaries_above,
            )
        )
        ne_features.append(
            dict(
                category='physical',
                name='ocean',
                scale=bnd_resolution,
                edgecolor='k',
                facecolor='none',
                zorder=2 + 10*boundaries_above,
            )
        )
        ne_features.append(
            dict(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale=bnd_resolution,
                edgecolor='grey',
                facecolor='none',
                zorder=2 + 10*boundaries_above,
            )
        )

        # land/sea mask
        if land_mask is True:
            ne_features.append(
                dict(
                    category='physical',
                    name='land',
                    scale=bnd_resolution,
                    edgecolor='k',
                    facecolor='w',
                    zorder=11,
                )
            )
        if sea_mask is True:
            ne_features.append(
                dict(
                    category='physical',
                    name='ocean',
                    scale=bnd_resolution,
                    edgecolor='k',
                    facecolor='w',
                    zorder=11,
                )
            )

    for f in ne_features:
        ax.add_feature(cfeature.NaturalEarthFeature(**f))

    # zoom
    if bbox is not None:
        ax.set_extent(bbox, crs=ccrs.PlateCarree())

    # grid
    if grid_resolution is not None:
        meridians = np.concatenate((
            np.arange(0, -181, -grid_resolution[0]),
            np.arange(0, 181, grid_resolution[0]),
        ))
        # if bbox and bbox[1] > 180:
        #     meridians = [
        #         m for m in meridians
        #         if m % 360 >= bbox[0] and m % 360 <= bbox[1]
        #     ]
        parallels = np.concatenate((
            np.arange(0, -100, -grid_resolution[1]),
            np.arange(0, 100, grid_resolution[1]),
        ))
        gl = ax.gridlines(
            xlocs=meridians,
            ylocs=parallels,
            draw_labels=isinstance(
                projection, (ccrs.PlateCarree, ccrs.Mercator)
            ),
            # linewidth=1,
            # color='k',
            # linestyle='--',
            # xlim=(bbox[2], bbox[3]),
            # ylim=(bbox[0], bbox[1]),
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    return fig, ax


def mpl_taylor(
        x_coords, y_coords, labels, colors, markers, refstd,
        crmse_levels, crmse_labels, scatter_kw={}, frame=False,
        legend=True, legend_type='fig', legend_kw={},
        fig=None, ax=None):
    """
    Draw a Line2D plot of a DataFrame columns.

    Parameters
    ----------
    x_coord, y_coord : 1D arrays
        Point coordinates.
    labels : list of str
        List of labels for the legend.
    colors : None or list of str
        Marker colors corresponding to each objects.
    markers : None or list of str
        Marker shapes corresponding to each objects.
    refstd : float
        Standard deviation reference value.
    crmse_levels : int
        Number of CRMSE arcs of circle.
    crmse_labels : bool
        If True, crmse values are displayed for each arc.
    scatter_kw : dict
        Keyword arguments passed to matplotlib.pyplot.scatter.
    frame : bool
        If false, top and right figure boundaries are not drawn.
    legend : bool
        If True, the legend is plotted.
    legend_type : str
        Set to 'fig' for figure legend or to 'ax' for axis legend.
    legend_kw : dict
        Keyword arguments passed to pyplot.legend or plt.figlegend.
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
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    ax.grid(False, which='both')

    # max of x and y axes
    smax = 1.6*refstd

    ax.set_xlim(0, smax)
    ax.set_ylim(0, smax)
    if refstd == 1:
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
    contours = ax.contour(
        xs, ys, rms, crmse_levels,
        colors='k', linestyles='dotted', linewidths=0.5,
    )
    if crmse_labels:
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
            plt.text(
                x+m*np.cos(ang), y+m*np.sin(ang), corr, color='b',
                rotation=(ang-np.pi/2)*180/np.pi,
                verticalalignment='center', horizontalalignment='center',
            )
    plt.text(
        smax*np.sqrt(2)/2, smax*np.sqrt(2)/2,
        tr.taylor_diagram['corr'][pl.LANG], color='b',
        rotation=-45, verticalalignment='center',
        horizontalalignment='center', size='x-large',
    )

    # draw correlation circles
    for r in [refstd, smax*(15/16.)]:
        theta = np.linspace(0, np.pi/2)
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        ax.plot(x, y, c='#FF3FFF', linestyle='-')

    # plot model points
    for x, y, l, m, c in zip(x_coords, y_coords, labels, markers, colors):
        ax.scatter(x, y, marker=m, c=c, label=l, zorder=10, **scatter_kw)

    if legend:
        if legend_type == 'fig':
            plt.figlegend(**legend_kw)
        elif legend_type == 'ax':
            plt.legend(**legend_kw)

    return fig, ax
