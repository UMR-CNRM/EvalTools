# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
This module is designed to compute some Fairmode metrics.

Documentation on the different metric can be found in the "FAIRMODE guidance
document on modelling quality objectives and benchmarking".

https://fairmode.jrc.ec.europa.eu/activity/ct2

"""
import numpy as np
import pandas as pd
from functools import wraps
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import evaltools as evt
from .evaluator import Evaluator
from .plotting._utils import plot_func
from ._deprecate import deprecate_kwarg, deprecate


def _add_method(cls):
    """Add a method to a class."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func
    return decorator


def _add_fairmode_method(func):
    """Add a method to the Evaluator class."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, '_fairmode_params'):
            raise evt.EvaltoolsError(
                f"Fairmode coefficients must be set before "
                f"calling {func.__name__}."
            )
        return func(*args, **kwargs)

    setattr(Evaluator, func.__name__, wrapper)

    return wrapper


@_add_method(Evaluator)
def set_fairmode_params(self, availability_ratio=.75):
    r"""
    Set Fairmode coefficients used to calculate the measurement uncertainty.

    The coefficients are

        threshold : scalar
            Limit concentration value fixed by air quality policies.
        U : scalar
            $U^{95}_{95,r}$ as defined by FAIRMODE for measurement
            uncertainty calculation.
        alpha : scalar
            $\alpha$ as defined by FAIRMODE for measurement
            uncertainty calculation.
        RV : scalar
            Reference value as defined by FAIRMODE for measurement
            uncertainty calculation.
        perc : scalar
            Selected percentile value used in the calculation of FAIRMODE's
            modeling perfomance criteria for high percentiles.
        Np, Nnp :
            Coefficients used to compute in FAIRMODE's observation uncertainty
            for annual averages.

    Parameters
    ----------
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.

    """
    if self.series_type != 'hourly':
        raise evt.EvaltoolsError(
            "Fairmode parameters can't be safely set if the object's "
            "attribute 'series_type' is not 'hourly'. You can however set "
            "the object attribute _fairmode_params manually with a dictionary "
            "that has the following keys: availability_ratio, obj, "
            "species_name, threshold, u95r, alpha, RV, perc, Np, Nnp and beta."
        )

    if self.species.lower() in ['pm10']:
        self._fairmode_params = dict(
            availability_ratio=availability_ratio,
            obj=self.daily_mean(availability_ratio),
            species_name=r'Surface PM10 aerosol daily mean [$\mu g/m^3$]',
            threshold=50.,
            u95r=0.28,
            alpha=0.25,
            RV=50,
            perc=0.901,
            Np=20.,
            Nnp=1.5,
            beta=2.,
        )
    elif self.species.lower() in ['pm2p5', 'pm2.5', 'pm25']:
        self._fairmode_params = dict(
            availability_ratio=availability_ratio,
            obj=self.daily_mean(availability_ratio),
            species_name=r'Surface PM2.5 aerosol daily mean [$\mu g/m^3$]',
            threshold=25,
            u95r=0.36,
            alpha=0.5,
            RV=25,
            perc=0.901,
            Np=20.,
            Nnp=1.5,
            beta=2.,
        )
    elif self.species.lower() in ['no2', 'no_2']:
        self._fairmode_params = dict(
            availability_ratio=availability_ratio,
            obj=self,
            species_name=r'Surface nitrogen dioxide hourly mean [$\mu g/m^3$]',
            threshold=200.,
            u95r=0.24,
            alpha=0.2,
            RV=200,
            perc=0.998,
            Np=5.2,
            Nnp=5.5,
            beta=2.,
        )
    elif self.species.lower() in ['o3', 'o_3']:
        self._fairmode_params = dict(
            availability_ratio=availability_ratio,
            obj=self.moving_average_daily_max(availability_ratio),
            species_name=(
                r'Surface ozone 8h moving average daily maximum [$\mu g/m^3$]'
            ),
            threshold=120.,
            u95r=0.18,
            alpha=0.79,
            RV=120,
            perc=0.929,
            Np=11.,
            Nnp=3.,
            beta=2.,
        )
    else:
        raise evt.EvaltoolsError(
            f"Species {self.species} not supported by Fairmode's metrics."
        )


@_add_fairmode_method
def rmsu(self, threshold=.75, forecast_day=0):
    """
    Calculate the root mean square of measurement uncertainty.

    Parameters
    ----------
    threshold : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.

    Returns
    -------
    pandas.Series
        Series with index corresponding to the object stations
        and containing root mean square of measurement uncertainty
        for each station.

    """
    obs = self._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
    u95r = self._fairmode_params['u95r']
    alpha = self._fairmode_params['alpha']
    rv = self._fairmode_params['RV']
    mu = u95r*np.sqrt((1-alpha**2)*obs**2+(alpha*rv)**2)
    res = evt.scores.stats2d(
        mu,
        pd.DataFrame(0., index=obs.index, columns=obs.columns),
        score_list=['RMSE'],
        axis=0,
        threshold=threshold,
        keep_nan=True,
    )
    res.columns = ['RMSU']
    return res.RMSU


@_add_fairmode_method
def mqi(self, threshold=.75, forecast_day=0):
    """
    Calculate the modelling quality indicator.

    Parameters
    ----------
    threshold : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.

    Returns
    -------
    pandas.Series
        Series with index corresponding to the object stations
        and containing modelling quality incator for each station.

    """
    obs = self._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
    sim = self._fairmode_params['obj'].get_sim(forecast_day=forecast_day)
    rmse = evt.scores.stats2d(
        obs, sim, score_list=['RMSE'], axis=0,
        threshold=threshold,
        keep_nan=True,
    )
    rmsu = self.rmsu(threshold=threshold, forecast_day=forecast_day)
    res = rmse.RMSE/(self._fairmode_params['beta']*rmsu)
    return res


@_add_fairmode_method
def mqi90(self, threshold=.75, forecast_day=0):
    """
    Calculate the 90th percentile of modelling quality indicator values.

    Parameters
    ----------
    threshold : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.

    Returns
    -------
    float
        90th percentile of modelling quality incator.

    """
    mqi = self.mqi(threshold=threshold, forecast_day=forecast_day)
    mqi = mqi.dropna().sort_values()
    if len(mqi) == 0:
        return np.nan
    if len(mqi) == 1:
        return mqi.iloc[0]*.9
    stat90 = int(len(mqi)*0.9)
    dist = len(mqi)*0.9 - stat90
    return mqi.iloc[stat90-1] + (mqi.iloc[stat90]-mqi.iloc[stat90-1])*dist


@_add_fairmode_method
def mqi_y(self, availability_ratio=.75, forecast_day=0):
    """
    Calculate the MQIs for the average of model values.

    The period over which to average the data should preferably be one year.

    Parameters
    ----------
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.

    Returns
    -------
    pandas.Series
        Series with index corresponding to the object stations
        and containing yearly modelling quality incator for each station.

    """
    obs = self._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
    sim = self._fairmode_params['obj'].get_sim(forecast_day=forecast_day)
    scores = evt.scores.stats2d(
        obs, sim,
        score_list=[
            'MeanBias',
            'obs_mean',
            'obs_percentile 0.9',
            'sim_percentile 0.9',
        ],
        axis=0,
        threshold=float(availability_ratio),
        keep_nan=True,
    )

    alpha = self._fairmode_params['alpha']
    n_p = self._fairmode_params['Np']
    rv = self._fairmode_params['RV']
    n_np = self._fairmode_params['Nnp']
    u95r = self._fairmode_params['u95r']
    beta = self._fairmode_params['beta']

    u95 = (1-alpha**2)*(scores['obs_mean']**2) / n_p
    u95 += (alpha**2)*(rv**2)/n_np
    u95 = u95r*np.sqrt(u95)

    mqi = scores['MeanBias'].abs()/(beta*u95)
    return mqi


@_add_fairmode_method
def y90(self, availability_ratio=.75, forecast_day=0):
    """
    Calculate the 90th percentile of MQIs for the average of model values.

    The period over which to average the data should preferably be one year.

    Parameters
    ----------
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day corresponding to the data used in the calculation.

    Returns
    -------
    float
        90th percentile of modelling quality incator
        for yearly average model results.

    """
    mqi = self.mqi_y(
        availability_ratio=availability_ratio,
        forecast_day=forecast_day,
    )
    mqi = mqi.dropna().sort_values()

    if len(mqi) == 0:
        return np.nan
    if len(mqi) == 1:
        return mqi.iloc[0]*.9

    stat90 = int(len(mqi)*0.9)
    dist = len(mqi)*0.9 - stat90

    return mqi.iloc[stat90-1] + (mqi.iloc[stat90]-mqi.iloc[stat90-1])*dist


@plt.rc_context({'figure.autolayout': False})
@plot_func
def plot_fairmode_summary(
        self, availability_ratio=.75, forecast_day=0, title=None, label=None,
        return_mpc=False, write_categories=True, fig=None, ax=None):
    """
    Summary statistics diagram.

    Assessement summary diagram as described in FAIRMODE guidance
    document on modelling quality objectives and benchmarking.

    Parameters
    ----------
    self : evaltools.Evaluator object
        Object used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    title : str
        Diagram title.
    label : str
        Label for the default title.
    write_categories : bool
        If True, write "observations", "time" and "space" on the left of the
        plot.

    """
    def common_params(ax, xmin, xmax, points, mqi=None, sym=True):
        """
        Draw common features to all subplots.

        Parameters
        ----------
        ax : matplotlib axis
            Current subplot.
        xmin, xmax : scalar
            Limit values for the plot
        points : 1D array-like
            Values for the scatter-plot.
        mqi : scalar
            Modeling quality indicator. If < 1 for 90% of the stations,
            mqo is fullfilled (green dot, otherwise red).
        sym : bool
            Must be set to True if the subplot statistical indicator
            can be negative.

        """
        # plot points
        ax.scatter(points, np.ones(len(points)), zorder=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(False, which='both')

        # if points live out of limits, plot a point in the dashed area
        if (points > xmax).any():
            ax.scatter(
                [xmax+(xmax-xmin)*0.04], [1],
                clip_on=False,
                c=dot_col,
            )
        if (points < xmin).any():
            ax.scatter(
                [xmin-(xmax-xmin)*0.04], [1],
                clip_on=False,
                c=dot_col,
            )

        # remove y ticks
        ax.tick_params(axis='y', which='both', left=False)
        ax.set_yticks([0])
        ax.set_yticklabels([""])

        # draw a dashed rectangle at the end of the domain
        if sym is True:
            ax.spines['left'].set_visible(False)
            pol = plt.Polygon(
                xy=[[xmin, 2],
                    [xmin-(xmax-xmin)*0.08, 2],
                    [xmin-(xmax-xmin)*0.08, 0],
                    [xmin, 0]],
                closed=False, ls='--',
                clip_on=False,
                fc='none',
                edgecolor='k',
            )
            ax.add_patch(pol)
        ax.spines['right'].set_visible(False)
        pol = plt.Polygon(
            xy=[[xmax, 2],
                [xmax+(xmax-xmin)*0.08, 2],
                [xmax+(xmax-xmin)*0.08, 0],
                [xmax, 0]],
            closed=False,
            ls='--',
            clip_on=False,
            fc='none',
            edgecolor='k',
        )
        ax.add_patch(pol)

        # reduce tick size
        ax.tick_params(axis='x', labelsize='small')

        # # give more space to y label
        # box = ax.get_position()
        # ax.set_position(
        #     [box.x0+box.width*0.15, box.y0, box.width*0.8, box.height*0.4]
        # )

        # MQI fulfillment (plot a green or red dot)
        if mqi is not None:
            mqo = np.sum(mqi < 1)/float(len(mqi)) >= 0.9
            col = '#4CFF00'*int(mqo) + 'r'*int(~mqo)
            ax.scatter(
                [xmax+(xmax-xmin)*0.15], [1],
                clip_on=False,
                c=col,
                s=100,
            )
            return mqo

    if not hasattr(self, 'fairmode_params'):
        self.set_fairmode_params(availability_ratio)

    if title is None:
        title = (
            "{model}\n" +
            "{spe}\n" +
            "{start_date} 00UTC to {end_date} 23UTC"
        ).format(
            model=label or self.model,
            spe=self._fairmode_params['species_name'],
            start_date=self.start_date,
            end_date=self.end_date,
        )

    # scores
    scores = self._fairmode_params['obj'].temporal_scores(
        score_list=[
            'MeanBias', 'PearsonR', 'obs_std', 'sim_std',
            'obs_mean', 'sim_mean',
        ],
        availability_ratio=availability_ratio)[f'D{forecast_day}']
    if scores.isna().all().all():
        print("No valid stations !!!")
        if return_mpc:
            return None, None, None
        else:
            return None, None

    rmsu = self.rmsu(
        threshold=float(availability_ratio),
        forecast_day=forecast_day,
    )
    beta = self._fairmode_params['beta']
    perc = self._fairmode_params['perc']
    threshold = self._fairmode_params['threshold']
    u95r = self._fairmode_params['u95r']
    alpha = self._fairmode_params['alpha']
    n_p = self._fairmode_params['Np']
    rv = self._fairmode_params['RV']
    n_np = self._fairmode_params['Nnp']

    # plotting
    fig = fig or plt.figure(figsize=(9, 6))
    ax = ax or fig.add_subplot(1, 1, 1)
    ax.clear()
    ax.axis('off')

    # plt.subplots_adjust(left=.25, right=.85)

    plt.title(title)
    dot_col = "#1F77B4"
    ymin = 0
    ymax = 2
    mpc_dict = {}

    sub_axes = []
    n_sub_axes = 8
    lmarg = .18
    rmarg = .05
    for i in range(1, n_sub_axes+1):
        # sub_axes.append(
        #     ax.inset_axes(
        #         [0, marg + (i-2*marg*i)/n_sub_axes, 1, .04]
        #     )
        # )
        sub_axes.append(
            ax.inset_axes(
                [lmarg, 1 - i/n_sub_axes, 1 - lmarg - rmarg, .04]
            )
        )
        # sub_axes.append(ax.inset_axes([0, 1-i/n_sub_axes, 1, .04]))

    # ------- obs mean -------
    ax1 = sub_axes[0]
    common_params(
        ax1, xmin=0, xmax=100, points=scores['obs_mean'], sym=False,
    )
    ax1.set_ylabel(
        "Observed   \nmean ",
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
        size='small',
    )
    ax1.text(105, -2, r"$\mu gm^{-3}$")

    # ------- obs exceedences -------
    ax2 = sub_axes[1]
    valid_stations = scores.index[~scores[scores.keys()[0]].isna()]
    nb_exceedences = (
        self._fairmode_params['obj'].get_obs(forecast_day=forecast_day) >
        threshold
    )
    nb_exceedences = nb_exceedences[valid_stations].sum()
    common_params(ax2, xmin=0, xmax=100, points=nb_exceedences, sym=False)
    ax2.set_ylabel(
        "Observed   \nexceedences \n(> " +
        str(threshold) + r" $\mu gm^{-3}$)",
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
        size='small')
    ax2.text(105, -2, "days")

    # ------- TIME Bias Norm -------
    ax3 = sub_axes[2]
    x = scores['MeanBias']/(beta*rmsu)
    mpc = common_params(
        ax3, xmin=-2, xmax=2, points=x, mqi=np.abs(x).dropna(),
    )
    mpc_dict['time_bias'] = mpc
    ax3.set_ylabel(
        "Bias Norm",
        size='small',
        # "$\\frac{Mean\\:bias}{\\beta RMS_U}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center')
    # colored areas
    rect = plt.Rectangle(
        xy=(-1., 0.), width=2., height=2.,
        edgecolor='none', fc='#FFA500',
    )
    ax3.add_patch(rect)
    rect = plt.Rectangle(
        xy=(-.7, 0.), width=1.4, height=2.,
        edgecolor='none', fc='#4CFF00',
    )
    ax3.add_patch(rect)

    # ------- TIME Corr Norm -------
    ax4 = sub_axes[3]
    x = (2.*scores['obs_std']*scores['sim_std']*(
            1. - scores['PearsonR'])) / (beta*rmsu)**2
    mpc = common_params(
        ax4, xmin=0, xmax=2, points=x, mqi=x.dropna(), sym=False,
    )
    mpc_dict['time_corr'] = mpc
    ax4.set_ylabel(
        "1-R Norm",
        size='small',
        # "$\\frac{2\\sigma_O\\sigma_M(1-R)}{\\beta^2 RMS_U^2}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
    )
    # colored areas
    rect = plt.Rectangle(xy=(0., 0.), width=1., height=2.,
                         edgecolor='none', fc='#FFA500')
    ax4.add_patch(rect)
    rect = plt.Rectangle(xy=(0., 0.), width=.5, height=2.,
                         edgecolor='none', fc='#4CFF00')
    ax4.add_patch(rect)

    # ------- TIME StdDev Norm -------
    ax5 = sub_axes[4]
    x = (scores['sim_std']-scores['obs_std'])/(beta*rmsu)
    mpc = common_params(
        ax5, xmin=-2, xmax=2, points=x, mqi=np.abs(x).dropna(),
    )
    mpc_dict['time_std'] = mpc
    ax5.set_yticks([1.])
    ax5.set_ylabel(
        "StDev Norm",
        size='small',
        # "$\\frac{\\sigma_M-\\sigma_O}{\\beta RMS_U}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center')
    # colored areas
    rect = plt.Rectangle(
        xy=(-1., 0.), width=2., height=2.,
        edgecolor='none', fc='#FFA500',
    )
    ax5.add_patch(rect)
    rect = plt.Rectangle(
        xy=(-.7, 0.), width=1.4, height=2.,
        edgecolor='none', fc='#4CFF00',
    )
    ax5.add_patch(rect)

    # ------- Hperc -------
    ax6 = sub_axes[5]
    obs = self._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
    sim = self._fairmode_params['obj'].get_sim(forecast_day=forecast_day)
    perc_values = evt.scores.stats2d(
        obs, sim,
        score_list=[
            'obs_percentile {}'.format(perc),
            'sim_percentile {}'.format(perc),
        ],
        axis=0, threshold=float(availability_ratio),
        keep_nan=True,
    )
    u_95 = u95r*np.sqrt(
        (1-alpha**2) *
        (perc_values['obs_percentile {}'.format(perc)]**2) +
        (alpha**2)*(rv**2)
    )
    h_perc = (
        (
            perc_values['sim_percentile {}'.format(perc)] -
            perc_values['obs_percentile {}'.format(perc)]
        ) / (beta*u_95)
    )
    mpc = common_params(
        ax6, xmin=-2, xmax=2, points=h_perc, mqi=np.abs(h_perc).dropna(),
    )
    mpc_dict['time_hperc'] = mpc
    ax6.set_ylabel(
        "Hperc Norm",
        size='small',
        # "$\\frac{M_{perc}-O_{perc}}{\\beta U_{95}(O_{perc})}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center')
    # colored area
    rect = plt.Rectangle(xy=(-1., 0.), width=2., height=2.,
                         edgecolor='none', fc='#4CFF00')
    ax6.add_patch(rect)

    # ------- SPACE Corr Norm -------
    ax7 = sub_axes[6]
    sim = scores['sim_mean']
    obs = scores['obs_mean']
    corr = ((np.nanmean((obs-np.nanmean(obs))*(sim-np.nanmean(sim)))) /
            (np.nanstd(obs)*np.nanstd(sim)))
    u_95 = u95r*np.sqrt(
        (1-alpha**2)*(obs**2)/n_p + (alpha**2)*(rv**2)/n_np
    )
    rmsu_ = np.sqrt(np.nanmean(u_95**2))
    x = ((2.*np.nanstd(obs)*np.nanstd(sim)*(1. - corr)) /
         (beta*rmsu_)**2)
    mpc = common_params(
        ax7, xmin=0, xmax=2, points=np.array([x]), sym=False,
        mqi=np.array([x]),
    )
    mpc_dict['spatial_corr'] = mpc
    ax7.set_ylabel(
        "1-R Norm",
        size='small',
        # "$\\frac{2\\sigma_\\bar{O}\\sigma_\\bar{M}(1-R)}" +
        # "{\\beta^2 RMS_\\bar{U}^2}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
    )
    # colored areas
    rect = plt.Rectangle(
        xy=(0., 0.), width=1., height=2., edgecolor='none', fc='#FFA500',
    )
    ax7.add_patch(rect)
    rect = plt.Rectangle(
        xy=(0., 0.), width=.5, height=2., edgecolor='none', fc='#4CFF00',
    )
    ax7.add_patch(rect)

    # ------- SPACE StDev Norm -------
    ax8 = sub_axes[7]
    x = (np.nanstd(sim)-np.nanstd(obs))/(beta*rmsu_)
    mpc = common_params(
        ax8, xmin=-2, xmax=2, points=np.array([x]), mqi=np.abs(np.array([x])),
    )
    mpc_dict['spatial_std'] = mpc
    ax8.set_ylabel(
        "StDev Norm",
        size='small',
        # "$\\frac{\\sigma_\\bar{M}-\\sigma_\\bar{O}}" +
        # "{\\beta RMS_\\bar{U}}$",
        # size='large',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
    )
    ax8.annotate(
        '{valid}/{all} valid stations'.format(
            valid=len(valid_stations), all=len(self.stations)),
        xy=(1, 0), xycoords='axes fraction', fontsize='large',
        xytext=(40, -20), textcoords='offset points', ha='right',
        va='top',
    )
    # colored areas
    rect = plt.Rectangle(
        xy=(-1., 0.), width=2., height=2.,
        edgecolor='none', fc='#FFA500',
    )
    ax8.add_patch(rect)
    rect = plt.Rectangle(
        xy=(-.7, 0.), width=1.4, height=2.,
        edgecolor='none', fc='#4CFF00',
    )
    ax8.add_patch(rect)

    if write_categories:
        ax1.text(-37, -8, "-- observations --", rotation='vertical')
        ax4.text(
            -37/50., -13,
            "----------------- time -----------------",
            rotation='vertical',
        )
        ax7.text(-37/50., -8, "------ space ------", rotation='vertical')

    if return_mpc:
        return fig, ax, mpc_dict
    else:
        return fig, ax


@plot_func
def plot_target_diagram(
        obj, availability_ratio=.75, forecast_day=0,
        label=None, color=None, title=None, output_csv=None,
        list_stations_ouside_target=True, mark_by=None,
        indicative_color=False, return_mqi=False, fig=None, ax=None):
    """
    Plot the assessment target diagram.

    Assessement target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Object used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    label : str
        Label for the legend.
    color : None or str
        Point color.
    title : str
        Diagram title.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    list_stations_ouside_target : bool
        If True, codes of stations outside the target are written on the
        side of the graph. The option works when only one object is used
        for computation.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of obj.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.

    """
    if not hasattr(obj, '_fairmode_params'):
        obj.set_fairmode_params(availability_ratio)

    if label is None:
        label = obj.model
    if color is None:
        color = obj.color
    if title is None:
        title = (
            "{model}\n" +
            "{spe}\n" +
            "{start_date} 00UTC to {end_date} 23UTC"
        ).format(
            model=label,
            spe=obj._fairmode_params['species_name'],
            start_date=obj.start_date,
            end_date=obj.end_date,
        )

    res = _target_diagram_multi_models(
        [obj],
        forecast_day=forecast_day, labels=[label], colors=[color],
        title=title,
        output_csv=output_csv,
        list_stations_ouside_target=list_stations_ouside_target,
        mark_by=mark_by, indicative_color=indicative_color,
        return_mqi=return_mqi, fig=fig, ax=ax,
    )

    return res


def _target_diagram_multi_models(
        objects, forecast_day=0,
        labels=None, colors=None, title="",
        list_stations_ouside_target=True, mark_by=None,
        indicative_color=False, output_csv=None,
        return_mqi=False, fig=None, ax=None):
    """
    Plot the assessment target diagram.

    Assessement target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    labels : list of str
        List of labels for the legend (length of the list of labels must be
        equal to the length of the list of objects).
    colors : None or list of str
        Line colors corresponding to each objects.
    title : str
        Diagram title.
    list_stations_ouside_target : bool
        If True, codes of stations outside the target are written on the
        side of the graph. The option works when only one object is used
        for computation.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of obj.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    output_file : str
        File where to save the plot (without extension).
    file_formats : list of str
        List of file extensions.

    Returns
    -------
        Couple (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        | corresponding to the produced plot. Note that if the plot has been
        | shown in the user interface window, the figure and the axis will not
        | be usable again.

    """
    colors = colors or [obj.color for obj in objects]
    labels = labels or [obj.model for obj in objects]

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    # axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    major_ticks = np.arange(-2, 3, 1)
    minor_ticks = np.arange(-2, 2.1, 0.1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels([2, 1, 0, 1, 2])
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_aspect('equal')
    plt.xlabel("CRMSE / $\\beta RMS_U$")
    plt.ylabel("Mean Bias / $\\beta RMS_U$")
    plt.grid(False, which='both')

    # target lines
    for coords in [(-2, 2, -2, 2), (-2, 2, 2, -2)]:
        plt.plot([coords[0], coords[1]], [coords[2], coords[3]], 'k', lw=0.5)
    plt.text(
        -1.9, 0, "R", color='k', verticalalignment='center',
        horizontalalignment='left', size='medium',
    )
    plt.text(
        1.9, 0, "SD", color='k', verticalalignment='center',
        horizontalalignment='right', size='medium',
    )
    plt.text(
        0, 1.9, "Mean bias > 0", color='k', verticalalignment='top',
        horizontalalignment='center', size='medium',
    )
    plt.text(
        0, -1.9, "Mean bias < 0", color='k', verticalalignment='bottom',
        horizontalalignment='center', size='medium',
    )

    # target circles
    smax = 1
    levels = 2
    xs, ys = np.meshgrid(np.linspace(-smax, smax), np.linspace(-smax, smax))
    rms = np.sqrt(xs**2 + ys**2)
    ax.contour(
        xs, ys, rms, levels, colors='k',
        linestyles='dotted', linewidths=0.5,
    )
    circle = plt.Circle((0, 0), 1, color='#ECECEC', zorder=0)
    ax.add_artist(circle)

    # Make place on the side of the figure
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # scatter plot
    mqi_colors = []
    mqi_dict = {}
    for obj, c, lab in zip(objects, colors, labels):
        ar = obj._fairmode_params['availability_ratio']
        beta = obj._fairmode_params['beta']
        rmsu = obj.rmsu(
            threshold=float(ar), forecast_day=forecast_day,
        )
        obs = obj._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
        sim = obj._fairmode_params['obj'].get_sim(forecast_day=forecast_day)
        sc = evt.scores.stats2d(
            obs, sim,
            score_list=['MeanBias', 'CRMSE', 'PearsonR'],
            axis=0,
            threshold=float(ar),
            keep_nan=True,
        )
        if sc.isna().all().all():
            print("No valid station !!!")
            return None, None, None
        # avoid warning when calculating std of all nan vector
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', 'Degrees of freedom <= 0 for slice',
            )
            idx_both_nan = np.logical_or(sim.isna(), obs.isna())
            obs = obs.where(~idx_both_nan)
            sim = sim.where(~idx_both_nan)
            stdo = np.std(obs, axis=0)
            stds = np.std(sim, axis=0)
        signx = (np.abs(stdo - stds) /
                 ((stdo*stds*2*(1-sc.PearsonR))**0.5) > 1)*2 - 1
        x = (sc.CRMSE/(beta*rmsu))*signx
        y = sc.MeanBias/(beta*rmsu)
        mqi90 = str(
            round(
                obj.mqi90(
                    threshold=float(ar),
                    forecast_day=forecast_day,
                ),
                3,
            )
        )
        y90 = str(
            round(
                obj.y90(
                    availability_ratio=float(ar),
                    forecast_day=forecast_day,
                ),
                3,
            )
        )

        # define scatter plot markers
        if mark_by is None:
            markers = None
        else:
            # sc.dropna(inplace=True)
            markers = [
                mark_by[1][obj.stations[mark_by[0]][code]]
                for code in sc.index
            ]
            handles = [
                mlines.Line2D(
                    [], [],
                    color='grey',
                    marker=mark_by[1][key],
                    label=key,
                    linestyle='',
                )
                for key in mark_by[1]
            ]
            legend_markers = plt.legend(handles=handles, loc='upper left')
            legend_markers.set_zorder(12)
            ax.add_artist(legend_markers)

        if indicative_color:
            mqi_colors.append(
                'red'
                if (float(mqi90) > 1 or float(y90) > 1)
                else 'green'
            )
        mqi_dict[obj.model] = mqi90

        evt.plotting._mpl.mscatter(
            x, y, ax=ax, m=markers, marker='^', facecolors='none',
            edgecolors=c,
            label=lab+"\n($MQI_{90}$ = "+mqi90+", $Y_{90}$ = "+y90+")",
            zorder=10,
        )

        if output_csv is not None:
            radius = np.sqrt(x**2+y**2)
            csv_data = pd.DataFrame({'x': x, 'y': y, 'radius': radius})
            csv_data.sort_values('radius', inplace=True, ascending=False)
            csv_data.to_csv(
                output_csv.format(model=obj.model),
                sep=' ', na_rep='nan', float_format='%g', header=True,
                index=True,
            )

    # legend
    plt.text(
        2.1, 1.6,
        "$\\alpha$ = {}".format(objects[0]._fairmode_params['alpha']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 1.4,
        "$\\beta$ = {}".format(objects[0]._fairmode_params['beta']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 1.2,
        "RV = {}".format(objects[0]._fairmode_params['RV']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 1,
        "$U^{RV}_{95,r}$ = " + str(objects[0]._fairmode_params['u95r']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 0.5, "{} stations".format(np.sum(~x.isna())),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    # plt.text(2.1, 0.7, "Model : {0}".format(obj.model))
    # plt.text(2.1, 0.5, "Species : {0}".format(obj.species))
    # plt.text(2.1, 0.5, "Period : {0} - {1}".format(
    #     obj.start_date.strftime('%Y%m%d'), obj.end_date.strftime('%Y%m%d')))
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for text, col in zip(legend.get_texts(), mqi_colors):
        text.set_color(col)
    ax.add_artist(legend)

    if list_stations_ouside_target and len(objects) == 1:
        mqi = np.sqrt(x**2+y**2).sort_values(ascending=False)
        idx = mqi > 1
        out_sta = mqi.loc[idx].index
        sep = [" ", "\n"]

        plt.text(2.1, -.35,
                 "{nb} ".format(nb=len(out_sta)) +
                 "station" + "s"*(len(out_sta) not in [0, 1]) +
                 " with MQI > 1" +
                 (":\n"+''.join([a+sep[i % 2]
                                 for i, a in enumerate(out_sta[:24])])
                  )*(len(out_sta) > 0),
                 color='k', verticalalignment='top',
                 horizontalalignment='left', size='small')

    # title
    plt.title(title, loc='center')

    if return_mqi:
        return fig, ax, mqi_dict
    else:
        return fig, ax


@_add_method(Evaluator)
@deprecate_kwarg('forecastDay', 'forecast_day')
@deprecate_kwarg('targetFile', 'target_file', stacklevel=3)
@deprecate_kwarg('summaryFile', 'summary_file', stacklevel=4)
@deprecate_kwarg('targetTitle', 'target_title', stacklevel=3)
@deprecate_kwarg('summaryTitle', 'summary_title', stacklevel=4)
@deprecate_kwarg('outputCSV', 'output_csv', stacklevel=5)
def fairmode_benchmark(
        self, target_file=None, summary_file=None,
        output_csv=None, availability_ratio=.75, label=None,
        target_title=None, summary_title=None, color=None,
        file_formats=['png'], forecast_day=0, mark_by=None,
        indicative_color=False, output_indicators=None):
    """
    Plot FAIRMODE target and summary diagrams.

    Concentration values must be in µg/m^3. Supported species are 'o3',
    'no2', 'pm10' and 'pm2p5'.

    Parameters
    ----------
    target_file :  str or None
        File where to save the target diagram (without extension).
        If None, the figure is shown in a popping window.
    summary_file :  str or None
        File where to save the summary diagram (without extension).
        If None, the figure is shown in a popping window.
    output_csv : str or None
        File where to save the target data.
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    label : str
        Label for the legend.
    target_title : str
        Target diagram title.
    summary_title : str
        Summary diagram title.
    color : str
    file_formats : list of str
        List of file extensions.
    forecast_day : int
        Forecast day used to compute the two diagrams.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of self.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels for in the target plot are green if
        MQI90 < 1 and Y90 < 1 and else they are red.
    output_indicators : str or None
        File where to save the mqi90 and MPCs.

    """
    fig, ax, mqi90 = plot_target_diagram(
        self,
        availability_ratio=availability_ratio, forecast_day=forecast_day,
        label=label, color=color, title=target_title,
        output_file=target_file, file_formats=file_formats,
        output_csv=output_csv, list_stations_ouside_target=True,
        mark_by=mark_by, indicative_color=indicative_color,
        return_mqi=True,
    )
    fig, ax, mpc_dict = plot_fairmode_summary(
        self,
        availability_ratio=availability_ratio, forecast_day=forecast_day,
        output_file=summary_file, file_formats=file_formats,
        title=summary_title, return_mpc=True, label=label,
    )

    if output_indicators:
        csv_data = pd.DataFrame(mpc_dict, index=[self.model])
        csv_data['MQO'] = float(mqi90[self.model]) <= 1
        csv_data['MQI'] = float(mqi90[self.model])
        csv_data.to_csv(
            output_indicators.format(model=self.model),
            sep=' ', na_rep='nan', header=True,
            index=True,
        )


@plot_func
def plot_forecast_target_diagram(
        obj, thr=None, availability_ratio=.75, forecast_day=0,
        label=None, color=None, title=None, output_csv=None,
        list_stations_ouside_target=True, mark_by=None,
        indicative_color=False, return_mqi=False, fig=None, ax=None):
    """
    FAIRMODE forecast target diagram.

    Forecast forecast target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Object used for plotting.
    thr : scalar
        Threshold used to compute False Alarm (FA) and Missed Alarm (MA).
        If the  FA/MA ratio is < 1 then the station point is in the negative
        portion of the x axis, and if FA/MA >= 1 the station point is in the
        positive portion.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    label : str
        Label for the legend.
    color : None or str
        Point color.
    title : str
        Diagram title.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    list_stations_ouside_target : bool
        If True, codes of stations outside the target are written on the
        side of the graph. The option works when only one object is used
        for computation.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of self.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.

    """
    if not hasattr(obj, '_fairmode_params'):
        obj.set_fairmode_params(availability_ratio)

    if label is None:
        label = obj.model
    if color is None:
        color = obj.color
    if title is None:
        title = (
            "{model}\n" +
            "{spe}\n" +
            "{start_date} 00UTC to {end_date} 23UTC"
        ).format(
            model=label,
            spe=obj._fairmode_params['species_name'],
            start_date=obj.start_date,
            end_date=obj.end_date,
        )

    res = _forecast_target_diagram_multi_models(
        objects=[obj],
        thr=(thr or obj._fairmode_params['threshold']),
        forecast_day=forecast_day,
        labels=[label],
        colors=[color],
        title=title,
        output_csv=output_csv,
        list_stations_ouside_target=list_stations_ouside_target,
        mark_by=mark_by,
        indicative_color=indicative_color,
        return_mqi=return_mqi,
        fig=fig,
        ax=ax,
    )

    return res


def _forecast_target_diagram_multi_models(
        objects, thr, persistence=None, availability_ratio=0.75,
        forecast_day=0, labels=None, colors=None, title="", output_csv=None,
        list_stations_ouside_target=True, mark_by=None,
        indicative_color=False, return_mqi=False, fig=None, ax=None):
    """
    FAIRMODE forecast target diagram.

    Forecast forecast target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    thr : scalar
        Threshold used to compute False Alarm (FA) and Missed Alarm (MA).
        If the  FA/MA ratio is < 1 then the station point is in the negative
        portion of the x axis, and if FA/MA >= 1 the station point is in the
        positive portion.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    labels : list of str
        List of labels for the legend (length of the list of labels must be
        equal to the length of the list of objects).
    colors : None or list of str
        Line colors corresponding to each objects.
    title : str
        Diagram title.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    list_stations_ouside_target : bool
        If True, codes of stations outside the target are written on the
        side of the graph. The option works when only one object is used
        for computation.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of self.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.

    Returns
    -------
        Couple (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        | corresponding to the produced plot. Note that if the plot has been
        | shown in the user interface window, the figure and the axis will not
        | be usable again.

    """
    if colors is None:
        colors = [obj.color for obj in objects]
    if labels is None:
        labels = [obj.model for obj in objects]

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    # axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    major_ticks = np.arange(-2, 3, 1)
    minor_ticks = np.arange(-2, 2.1, 0.1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels([2, 1, 0, 1, 2])
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_aspect('equal')
    plt.xlabel("CRMSE / $RMSE\\_persistence$")
    plt.ylabel("Mean Bias / $RMSE\\_persistence$")
    plt.grid(False, which='both')

    # target lines
    for coords in [(-2, 2, -2, 2), (-2, 2, 2, -2)]:
        plt.plot([coords[0], coords[1]], [coords[2], coords[3]], 'k', lw=0.5)
    plt.axvline(x=0, color='k', lw=0.5)

    # target circles
    smax = 1
    levels = 2
    xs, ys = np.meshgrid(np.linspace(-smax, smax), np.linspace(-smax, smax))
    rms = np.sqrt(xs**2 + ys**2)
    ax.contour(
        xs, ys, rms, levels, colors='k',
        linestyles='dotted', linewidths=0.5,
    )
    circle = plt.Circle((0, 0), 1, color='#ECECEC', zorder=0)
    ax.add_artist(circle)

    # Make place on the side of the figure
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # scatter plot
    mqi_colors = []
    mqi_dict = {}
    for o, c, lab in zip(objects, colors, labels):
        obj = o._fairmode_params['obj']

        if persistence is None:
            obj_pers = obj.observations.persistence_model()
        else:
            obj_pers = persistence

        obs_pers = obj_pers.get_obs(forecast_day=forecast_day)
        sim_pers = obj_pers.get_sim(forecast_day=forecast_day)
        u95r = o._fairmode_params['u95r']
        alpha = o._fairmode_params['alpha']
        rv = o._fairmode_params['RV']
        mu = u95r*np.sqrt((1-alpha**2)*sim_pers**2+(alpha*rv)**2)
        sc_pers = evt.scores.stats2d(
            np.maximum(
                abs(obs_pers + mu - sim_pers),
                abs(obs_pers - mu - sim_pers),
            ),
            pd.DataFrame(0., index=obs_pers.index, columns=obs_pers.columns),
            score_list=['RMSE'],
            axis=0,
            threshold=float(availability_ratio),
            keep_nan=True,
        )

        sub_obj = obj.sub_period(obj_pers.start_date, obj_pers.end_date)

        obs = sub_obj.get_obs(forecast_day=forecast_day)
        sim = sub_obj.get_sim(forecast_day=forecast_day)

        idx_both_nan = np.logical_or(sim_pers.isna(), sim.isna())
        sim_pers = sim_pers.where(~idx_both_nan)
        sim = sim.where(~idx_both_nan)

        sc = evt.scores.stats2d(
            obs, sim,
            score_list=['MeanBias', 'CRMSE', 'PearsonR', 'RMSE'],
            axis=0,
            threshold=float(availability_ratio),
            keep_nan=True,
        )

        if sc.isna().all().all():
            print("No valid station !!!")
            if return_mqi:
                return None, None, None
            else:
                return None, None

        contingency_table = [
            evt.scores.contingency_table(o_it[1], s_it[1], thr=thr)
            for o_it, s_it in zip(obs.items(), sim.items())
        ]
        fa = np.array(
            [contingency_table[i][1, 0] for i in range(len(contingency_table))]
        )
        ma = np.array(
            [contingency_table[i][0, 1] for i in range(len(contingency_table))]
        )

        if o is objects[0]:
            if len(objects) == 1 and fa.sum() == ma.sum() == 0:
                left_txt = ""
                right_txt = "FA = MA = 0"
                operator = np.ndarray.__ge__
            else:
                if fa.sum() >= ma.sum():
                    left_txt = "FA < MA"
                    right_txt = "FA $\\geqslant$ MA"
                    operator = np.ndarray.__ge__
                else:
                    left_txt = "FA $\\leqslant$ MA"
                    right_txt = "FA > MA"
                    operator = np.ndarray.__gt__
            plt.text(
                -1.9, 0, left_txt,
                color='k', verticalalignment='center',
                horizontalalignment='left', size='small',
            )
            plt.text(
                1.9, 0, right_txt,
                color='k', verticalalignment='center',
                horizontalalignment='right', size='small',
            )

        signx = operator(fa, ma)*2 - 1

        x = (sc.CRMSE/(sc_pers['RMSE']))*signx
        y = sc.MeanBias/(sc_pers['RMSE'])

        mqi = sc.RMSE/sc_pers['RMSE']

        def get_90perc_value(x):
            x = x.dropna().sort_values()
            if len(x) == 0:
                return np.nan
            if len(x) == 1:
                return x.iloc[0]*.9
            stat90 = int(len(x)*0.9)
            dist = len(x)*0.9 - stat90
            return x.iloc[stat90-1] + (x.iloc[stat90]-x.iloc[stat90-1])*dist

        mqi90 = str(round(get_90perc_value(mqi), 3))

        # define scatter plot markers
        if mark_by is None:
            markers = None
        else:
            # sc.dropna(inplace=True)
            markers = [
                mark_by[1][obj.stations[mark_by[0]][code]]
                for code in sc.index
            ]
            handles = [
                mlines.Line2D(
                    [], [], color='grey',
                    marker=mark_by[1][key],
                    label=key, linestyle='',
                )
                for key in mark_by[1]
            ]
            legend_markers = plt.legend(handles=handles, loc='upper left')
            legend_markers.set_zorder(12)
            ax.add_artist(legend_markers)

        if indicative_color:
            mqi_colors.append(
                'red'
                if float(mqi90) > 1
                else 'green'
            )
        mqi_dict[obj.model] = mqi90

        evt.plotting._mpl.mscatter(
            x, y, ax=ax, m=markers, marker='^', facecolors='none',
            edgecolors=c,
            label=lab+"\n($MQI_{90}$ = "+mqi90+")",
            zorder=10,
        )

        if output_csv is not None:
            radius = np.sqrt(x**2+y**2)
            csv_data = pd.DataFrame({'x': x, 'y': y, 'radius': radius})
            csv_data.sort_values('radius', inplace=True, ascending=False)
            csv_data.to_csv(
                output_csv.format(model=obj.model),
                sep=' ', na_rep='nan', float_format='%g', header=True,
                index=True,
            )

    plt.text(
        2.1, 1.6,
        f"Threshold = {thr}",
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 1.4,
        f"Forecast day = {forecast_day}",
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    plt.text(
        2.1, 0.9,
        "{}/{} valid stations".format(np.sum(~x.isna()), len(x)),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
    )
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for text, col in zip(legend.get_texts(), mqi_colors):
        text.set_color(col)
    ax.add_artist(legend)

    if list_stations_ouside_target and len(objects) == 1:
        mqi = np.sqrt(x**2+y**2).sort_values(ascending=False)
        idx = mqi > 1
        out_sta = mqi.loc[idx].index
        sep = [" ", "\n"]

        plt.text(
            2.1, -.35,
            "{nb} ".format(nb=len(out_sta)) +
            "station" + "s"*(len(out_sta) not in [0, 1]) +
            " with MQI > 1" +
            (
                ":\n"+''.join(
                    [a+sep[i % 2] for i, a in enumerate(out_sta[:24])]
                )
            )*(len(out_sta) > 0),
            color='k',
            verticalalignment='top',
            horizontalalignment='left',
            size='small',
        )

    # titles
    plt.title(title, loc='center')

    if return_mqi:
        return fig, ax, mqi_dict
    else:
        return fig, ax


setattr(
    Evaluator,
    'fairmodeBenchmark',
    deprecate('fairmodeBenchmark', fairmode_benchmark),
)


@plt.rc_context({'figure.autolayout': False})
@plot_func
def plot_yearly_fairmode_summary(
        self, availability_ratio=.75, forecast_day=0, title=None, label=None,
        return_mpc=False, write_categories=True, fig=None, ax=None):
    """
    Summary statistics diagram.

    Assessement summary diagram as described in FAIRMODE guidance
    document on modelling quality objectives and benchmarking.

    Parameters
    ----------
    self : evaltools.Evaluator object
        Object used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per
        forecast day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    title : str
        Diagram title.
    label : str
        Label for the default title.
    write_categories : bool
        If True, write "observations", "time" and "space" on the left of the
        plot.

    """
    def common_params(ax, xmin, xmax, points, mqi=None, sym=True):
        """
        Draw common features to all subplots.

        Parameters
        ----------
        ax : matplotlib axis
            Current subplot.
        xmin, xmax : scalar
            Limit values for the plot
        points : 1D array-like
            Values for the scatter-plot.
        mqi : scalar
            Modeling quality indicator. If < 1 for 90% of the stations,
            mqo is fullfilled (green dot, otherwise red).
        sym : bool
            Must be set to True if the subplot statistical indicator
            can be negative.

        """
        # plot points
        ax.scatter(points, np.ones(len(points)), zorder=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(False, which='both')

        # if points live out of limits, plot a point in the dashed area
        if (points > xmax).any():
            ax.scatter(
                [xmax+(xmax-xmin)*0.04], [1],
                clip_on=False,
                c=dot_col,
            )
        if (points < xmin).any():
            ax.scatter(
                [xmin-(xmax-xmin)*0.04], [1],
                clip_on=False,
                c=dot_col,
            )

        # remove y ticks
        ax.tick_params(axis='y', which='both', left=False)
        ax.set_yticks([0])
        ax.set_yticklabels([""])

        # draw a dashed rectangle at the end of the domain
        if sym is True:
            ax.spines['left'].set_visible(False)
            pol = plt.Polygon(
                xy=[[xmin, 2],
                    [xmin-(xmax-xmin)*0.08, 2],
                    [xmin-(xmax-xmin)*0.08, 0],
                    [xmin, 0]],
                closed=False, ls='--',
                clip_on=False,
                fc='none',
                edgecolor='k',
            )
            ax.add_patch(pol)
        ax.spines['right'].set_visible(False)
        pol = plt.Polygon(
            xy=[[xmax, 2],
                [xmax+(xmax-xmin)*0.08, 2],
                [xmax+(xmax-xmin)*0.08, 0],
                [xmax, 0]],
            closed=False,
            ls='--',
            clip_on=False,
            fc='none',
            edgecolor='k',
        )
        ax.add_patch(pol)

        # reduce tick size
        ax.tick_params(axis='x', labelsize='small')

        # # give more space to y label
        # box = ax.get_position()
        # ax.set_position(
        #     [box.x0+box.width*0.15, box.y0, box.width*0.8, box.height*0.4]
        # )

        # MQI fulfillment (plot a green or red dot)
        if mqi is not None:
            mqo = np.sum(mqi < 1)/float(len(mqi)) >= 0.9
            col = '#4CFF00'*int(mqo) + 'r'*int(~mqo)
            ax.scatter(
                [xmax+(xmax-xmin)*0.15], [1],
                clip_on=False,
                c=col,
                s=100,
            )
            return mqo

    if not hasattr(self, 'fairmode_params'):
        self.set_fairmode_params(availability_ratio)

    if title is None:
        title = (
            "{model}\n" +
            "{spe}\n" +
            "{startDate} 00UTC to {endDate} 23UTC"
        ).format(
            model=label or self.model,
            spe=self._fairmode_params['species_name'],
            startDate=self.startDate,
            endDate=self.endDate,
        )

    # scores
    scores = self._fairmode_params['obj'].temporal_scores(
        score_list=[
            'MeanBias', 'PearsonR', 'obs_std', 'sim_std',
            'obs_mean', 'sim_mean',
        ],
        availability_ratio=availability_ratio)[f'D{forecast_day}']
    if scores.isna().all().all():
        print("No valid stations !!!")
        if return_mpc:
            return None, None, None
        else:
            return None, None

    beta = self._fairmode_params['beta']
    u95r = self._fairmode_params['u95r']
    alpha = self._fairmode_params['alpha']
    n_p = self._fairmode_params['Np']
    rv = self._fairmode_params['RV']
    n_np = self._fairmode_params['Nnp']

    # plotting
    fig = fig or plt.figure(figsize=(9, 4))
    ax = ax or fig.add_subplot(1, 1, 1)
    ax.clear()
    ax.axis('off')

    # plt.subplots_adjust(left=.25, right=.85)

    plt.title(title)
    dot_col = "#1F77B4"
    ymin = 0
    ymax = 2
    mpc_dict = {}

    sub_axes = []
    n_sub_axes = 4
    lmarg = .18
    rmarg = .13
    for i in range(1, n_sub_axes+1):
        sub_axes.append(
            ax.inset_axes(
                [lmarg, 1 - i/n_sub_axes, 1 - lmarg - rmarg, 0.32/n_sub_axes]
            )
        )

    valid_stations = scores.index[~scores[scores.keys()[0]].isna()]

    sim = scores['sim_mean']
    obs = scores['obs_mean']
    corr = ((np.nanmean((obs-np.nanmean(obs))*(sim-np.nanmean(sim)))) /
            (np.nanstd(obs)*np.nanstd(sim)))
    u_95 = u95r*np.sqrt(
        (1-alpha**2)*(obs**2)/n_p + (alpha**2)*(rv**2)/n_np
    )
    rmsu_ = np.sqrt(np.nanmean(u_95**2))

    # ------- obs mean -------
    ax1 = sub_axes[0]
    print('obs mean')
    print(obs)
    common_params(
        ax1, xmin=0, xmax=100, points=obs, sym=False,
    )
    ax1.set_ylabel(
        "Observed   \nmean ",
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
        size='small',
    )
    ax1.text(105, -2, r"$\mu gm^{-3}$")

    # ------- TIME Bias Norm -------
    ax3 = sub_axes[1]
    x = scores['MeanBias']/(beta*rmsu_)
    print('TIME Bias Norm')
    print(x)
    mpc = common_params(
        ax3, xmin=-2, xmax=2, points=x, mqi=np.abs(x).dropna(),
    )
    mpc_dict['time_bias'] = mpc
    ax3.set_ylabel(
        "Bias Norm",
        size='small',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center')
    # colored areas
    # rect = plt.Rectangle(
    #    xy=(-1., 0.), width=2., height=2.,
    #    edgecolor='none', fc='#FFA500',
    # )
    # ax3.add_patch(rect)
    rect = plt.Rectangle(
        xy=(-1., 0.), width=2., height=2.,
        edgecolor='none', fc='#4CFF00',
    )
    ax3.add_patch(rect)

    # ------- SPACE Corr Norm -------
    ax7 = sub_axes[2]
    x = ((2.*np.nanstd(obs)*np.nanstd(sim)*(1. - corr)) /
         (beta*rmsu_)**2)
    print('SPACE Corr Norm')
    print(x)
    mpc = common_params(
        ax7, xmin=0, xmax=2, points=np.array([x]), sym=False,
        mqi=np.array([x]),
    )
    mpc_dict['spatial_corr'] = mpc
    ax7.set_ylabel(
        "1-R Norm",
        size='small',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
    )
    # colored areas
    rect = plt.Rectangle(
        xy=(0., 0.), width=1., height=2., edgecolor='none', fc='#FFA500',
    )
    ax7.add_patch(rect)
    rect = plt.Rectangle(
        xy=(0., 0.), width=.7, height=2., edgecolor='none', fc='#4CFF00',
    )
    ax7.add_patch(rect)

    # ------- SPACE StDev Norm -------
    ax8 = sub_axes[3]
    x = (np.nanstd(sim)-np.nanstd(obs))/(beta*rmsu_)
    print('SPACE StDev Norm')
    print(x)
    mpc = common_params(
        ax8, xmin=-2, xmax=2, points=np.array([x]), mqi=np.abs(np.array([x])),
    )
    mpc_dict['spatial_std'] = mpc
    ax8.set_ylabel(
        "StDev Norm",
        size='small',
        labelpad=70,
        rotation='horizontal',
        verticalalignment='center',
    )
    ax8.annotate(
        '{valid}/{all} valid stations'.format(
            valid=len(valid_stations), all=len(self.stations)),
        xy=(1, 0), xycoords='axes fraction', fontsize='large',
        xytext=(40, -20), textcoords='offset points', ha='right',
        va='top',
    )
    # colored areas
    rect = plt.Rectangle(
        xy=(-1., 0.), width=2., height=2.,
        edgecolor='none', fc='#FFA500',
    )
    ax8.add_patch(rect)
    rect = plt.Rectangle(
        xy=(-.7, 0.), width=1.4, height=2.,
        edgecolor='none', fc='#4CFF00',
    )
    ax8.add_patch(rect)

    if write_categories:
        ax1.text(-37, -1.3, "-- obs --", rotation='vertical')
        ax3.text(-37/25-2, -1.3, "-- time --", rotation='vertical')
        ax7.text(-37/50., -6, "------ space ------", rotation='vertical')

    if return_mpc:
        return fig, ax, mpc_dict
    else:
        return fig, ax


@plt.rc_context({"savefig.bbox": 'tight'})
@plot_func
def plot_scatter_diagram(
        obj, availability_ratio=.75, forecast_day=0,
        label=None, color=None, title=None, output_csv=None,
        mark_by=None,
        indicative_color=False, return_mqi=False, fig=None, ax=None):
    """
    Plot the assessment target diagram.

    Assessement target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    obj : evaltools.Evaluator object
        Object used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    label : str
        Label for the legend.
    color : None or str
        Point color.
    title : str
        Diagram title.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of obj.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.

    """
    if not hasattr(obj, '_fairmode_params'):
        obj.set_fairmode_params(availability_ratio)

    if label is None:
        label = obj.model
    if color is None:
        color = obj.color
    if title is None:
        title = (
            "{model}\n" +
            "{spe}\n" +
            "{startDate} 00UTC to {endDate} 23UTC"
        ).format(
            model=label,
            spe=obj._fairmode_params['species_name'],
            startDate=obj.startDate,
            endDate=obj.endDate,
        )

    res = _scatter_diagram_multi_models(
        [obj],
        forecast_day=forecast_day, labels=[label], colors=[color],
        title=title,
        output_csv=output_csv,
        mark_by=mark_by, indicative_color=indicative_color,
        return_mqi=return_mqi, fig=fig, ax=ax,
    )

    return res


def _fill_diag(obj, obsmean, max_val):
    alpha = obj._fairmode_params['alpha']
    n_p = obj._fairmode_params['Np']
    rv = obj._fairmode_params['RV']
    n_np = obj._fairmode_params['Nnp']
    u95r = obj._fairmode_params['u95r']
    beta = obj._fairmode_params['beta']

    xs = np.array([0, max_val])
    # equation (7) from guide v7.3
    u95 = (1-alpha**2)*(xs**2) / n_p
    u95 += (alpha**2)*(rv**2)/n_np
    u95 = u95r*np.sqrt(u95)

    # equation (6) from guide v7.3
    y1 = xs + beta * u95
    y2 = xs - beta * u95

    diagfill = plt.fill_between(
        xs,
        y1,
        y2,
        color='yellowgreen',
        edgecolor='k',
    )
    diag = plt.axline((0, 0), slope=1, color='k', lw=0.3)
    return [diag, diagfill]


def _scatter_diagram_multi_models(
        objects, availability_ratio=0.75, forecast_day=0,
        labels=None, colors=None, title="",
        mark_by=None,
        indicative_color=False, output_csv=None,
        return_mqi=False, fig=None, ax=None):
    """
    Plot the assessment target diagram.

    Assessement target diagram as described in FAIRMODE guidance document
    on modelling quality objectives and benchmarking.

    Parameters
    ----------
    objects : list of evaltools.Evaluator objects
        Evaluator objects used for plotting.
    availability_ratio : float
        Minimal rate of data available on the period required per forecast
        day to compute the scores for each station.
    forecast_day : int
        Forecast day used in the diagram.
    labels : list of str
        List of labels for the legend (length of the list of labels must be
        equal to the length of the list of objects).
    colors : None or list of str
        Line colors corresponding to each objects.
    title : str
        Diagram title.
    mark_by : 1D array-like
        This argument allows to choose different markers for different
        station groups according to a variable of obj.stations.
        It must be of length two. First element is the label of the column
        used to define the markers. Second element is a dictionary defining
        which marker to use for each possible values.
        Ex: ('area', {'urb': 's', 'rur': 'o', 'sub': '^'})
    indicative_color : bool
        If True, legend labels are green if MQI90 < 1 and Y90 < 1 and
        else they are red.
    output_csv : str or None
        File where to save the data. The File name must contain {model}
        instead of the model name (so that one file is written for each
        object showed on the graph).
    output_file : str
        File where to save the plot (without extension).
    file_formats : list of str
        List of file extensions.

    Returns
    -------
        Couple (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
        | corresponding to the produced plot. Note that if the plot has been
        | shown in the user interface window, the figure and the axis will not
        | be usable again.

    """
    colors = colors or [obj.color for obj in objects]
    labels = labels or [obj.model for obj in objects]

    # scores
    scores = objects[0]._fairmode_params['obj'].temporal_scores(
        score_list=[
            'MeanBias', 'PearsonR', 'obs_std', 'sim_std',
            'obs_mean', 'sim_mean',
        ],
        availability_ratio=availability_ratio)[f'D{forecast_day}']
    if scores.isna().all().all():
        print("No valid stations !!!")
        if return_mqi:
            return None, None, None
        else:
            return None, None

    sim = scores['sim_mean']
    obs = scores['obs_mean']
    max_val = max([np.nanmax(sim), np.nanmax(obs)]) * 1.05
    print(max_val)
    for i in range(10):
        if max_val % 10 != 0:
            max_val += 1
    print(max_val)

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(1, 1, 1)

    # axes
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.xlabel(r"OBS $\mu g/m^3$")
    plt.ylabel(r"MOD $\mu g/m^3$")
    plt.grid(False, which='both')

    # Green diagonal
    for diag in _fill_diag(objects[0], obs.mean(), max_val):
        ax.add_artist(diag)

    # scatter plot
    mqi_colors = []
    mqi_dict = {}
    for obj, c, lab in zip(objects, colors, labels):
        ar = obj._fairmode_params['availability_ratio']
        # beta = obj._fairmode_params['beta']
        # rmsu = obj.rmsu(
        #     threshold=float(ar), forecast_day=forecast_day,
        # )
        obs = obj._fairmode_params['obj'].get_obs(forecast_day=forecast_day)
        sim = obj._fairmode_params['obj'].get_sim(forecast_day=forecast_day)
        sc = evt.scores.stats2d(
            obs, sim,
            score_list=[
                'MeanBias', 'CRMSE', 'PearsonR', 'sim_mean', 'obs_mean',
            ],
            axis=0,
            threshold=float(ar),
            keep_nan=True,
        )
        print(sc)
        print(obj.get_obs(0).mean(axis=0))
        if sc.isna().all().all():
            print("No valid station !!!")
            return None, None, None
        x = sc.obs_mean  # (sc.CRMSE/(beta*rmsu))*signx
        y = sc.sim_mean  # sc.MeanBias/(beta*rmsu)
        mqi_y = obj.mqi_y(
            availability_ratio=float(ar),
            forecast_day=forecast_day,
        )
        print(sc.obs_mean)
        print(sc.MeanBias / mqi_y)
        diag_pos = 0.9 * x  # beta*rmsu / sc.obs_mean
        print(diag_pos)
        y90 = str(
            round(
                obj.y90(
                    availability_ratio=float(ar),
                    forecast_day=forecast_day,
                ),
                3,
            )
        )

        # define scatter plot markers
        if mark_by is None:
            markers = None
        else:
            markers = [
                mark_by[1][obj.stations[mark_by[0]][code]]
                for code in sc.index
            ]
            handles = [
                mlines.Line2D(
                    [], [],
                    color='grey',
                    marker=mark_by[1][key],
                    label=key,
                    linestyle='',
                )
                for key in mark_by[1]
            ]
            legend_markers = plt.legend(handles=handles, loc='upper left')
            legend_markers.set_zorder(12)
            ax.add_artist(legend_markers)

        if indicative_color:
            mqi_colors.append(
                'red'
                if float(y90) > 1
                else 'green'
            )
        mqi_dict[obj.model] = {'y90': y90}

        evt.plotting._mpl.mscatter(
            x, y, ax=ax, m=markers, marker='^', facecolors='none',
            edgecolors=c,
            label=lab+"\n($Y_{90}$ = "+y90+")",
            zorder=10,
        )

        if output_csv is not None:
            csv_data = pd.DataFrame({'x': x, 'y': y})
            csv_data.to_csv(
                output_csv.format(model=obj.model),
                sep=' ', na_rep='nan', float_format='%g', header=True,
                index=True,
            )

    # legend
    x_pos = 1.01
    plt.text(
        x_pos, 0.99,
        "$\\alpha$ = {}".format(objects[0]._fairmode_params['alpha']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )
    plt.text(
        x_pos, 0.94,
        "$\\beta$ = {}".format(objects[0]._fairmode_params['beta']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )
    plt.text(
        x_pos, 0.90,
        "RV = {}".format(objects[0]._fairmode_params['RV']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )
    plt.text(
        x_pos, 0.85,
        "$U^{RV}_{95,r}$ = " + str(objects[0]._fairmode_params['u95r']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )
    plt.text(
        x_pos, 0.81,
        "$N_{p}$ = " + str(objects[0]._fairmode_params['Np']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )
    plt.text(
        x_pos, 0.77,
        "$N_{np}$ = " + str(objects[0]._fairmode_params['Nnp']),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )

    plt.text(
        x_pos, 0.65,
        "{} stations".format(np.sum(~x.isna())),
        color='k',
        verticalalignment='center',
        horizontalalignment='left',
        size='medium',
        transform=ax.transAxes,
    )

    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for text, col in zip(legend.get_texts(), mqi_colors):
        text.set_color(col)
    ax.add_artist(legend)

    # title
    plt.title(title, loc='center')

    if return_mqi:
        return fig, ax, mqi_dict
    else:
        return fig, ax
