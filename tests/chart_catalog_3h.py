"""Run all plotting functions using data with a 3h time step."""
import os
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

import evaltools as evt

###################################
# read consol and namelist params #
###################################

model_list = ['model1', 'model2', 'model3']
sc = 'RMSE'

#################
# read listings #
#################

listing_path = "../doc/sample_data/listing"
stations = evt.utils.read_listing(
    listing_path, classes='all', types='all', area_coord=None,
)

##########
# params #
##########

model_colors = {
    'model1': '#00FFFF',
    'model2': '#3333FF',
    'model3': '#9E3BCF',
}
model_markers = {
    'model1': '.',  # 'v',
    'model2': '.',  # '^',
    'model3': '.',  # 'D',
}
model_names = {
    'model1': 'ENSEMBLE MEDIAN',
    'model2': 'MOCAGE',
    'model3': 'CHIMERE',
}
species_names = {
    'o3': 'Surface ozone',
    'no2': 'Surface nitrogen dioxide',
    'pm10': 'Surface PM10 aerosol',
    'pm2p5': 'Surface PM2.5 aerosol',
    'co': 'Surface carbon monoxide',
}
score_names = {
    'RMSE': 'Root mean square error',
    'MMB': 'Modified mean bias',
    'PearsonR': 'Correlation',
    'Bias': 'Mean bias',
    'FracBias': 'Fractional bias',
}
colors = [model_colors[model] for model in model_list]
os.makedirs("../doc/source/charts", exist_ok=True)

##########
# charts #
##########

# objects building
obs_path = "../doc/sample_data/observations/{{year}}_{species}_{{station}}"
sim_path = ("../doc/sample_data/{model}{runtype}/J{{forecast_day}}/" +
            "{{year}}_{species}_{{station}}")
start_date = date(2017, 6, 1)
end_date = date(2017, 6, 4)
model_list = ["ENS", "MFM"]
model_colors = {"ENS": '#CB025A', "MFM": '#004D8A'}
species = 'no2'

observations = evt.evaluator.Observations.from_time_series(
    generic_file_path=obs_path.format(species=species),
    correc_unit=1e9,
    species=species,
    start=start_date,
    end=end_date,
    stations=stations,
    forecast_horizon=4,
    step=3,
)

objs = {}
daily_objs = {}
fh = 4
for model in model_list:
    generic_file_path = sim_path.format(
        model=model,
        runtype='forecast',
        species=species,
    )
    stations_idx = observations.stations.index
    simulations = evt.evaluator.Simulations.from_time_series(
        generic_file_path=generic_file_path,
        stations_idx=stations_idx,
        correc_unit=1,
        species=species,
        model=model,
        start=start_date,
        end=end_date,
        forecast_horizon=4,
        step=3,
    )
    objs[model] = evt.evaluator.Evaluator(observations, simulations,
                                          color=model_colors[model])
    daily_objs[model] = objs[model].daily_max()

print("time_series")
evt.plotting.plot_time_series(
    list(objs.values()),
    # station_list=stations.index[:3],
    colors=['#CB025A', '#004D8A'],
    start_end=[start_date, start_date+timedelta(days=3)],
    title="Time series of observed and modelled data\n\n",
    file_formats=['png'], black_axes=True, nb_of_minor_ticks=(7, 1),
    xticking='daily',
    output_file="../doc/source/charts/time_series_3h",
)
evt.plotting.plot_time_series(
    list(objs.values()), station_list=stations.index[:3],
    colors=['#CB025A', '#004D8A'],
    start_end=[start_date, start_date+timedelta(days=3)],
    title="Time series of observed and modelled data\n\n",
    file_formats=['png'], plot_type='median', envelope=True, black_axes=True,
    output_file="../doc/source/charts/time_series_quartiles_3h",
)

print("average_ft_scores")
evt.plotting.plot_average_ft_scores(
    list(objs.values()),
    score='rmse',
    labels=['ENSEMBLE', 'MOCAGE'],
    title="Temporal mean of spatial RMSE as a function of the forecast term\n",
    black_axes=True,
    outlier_thresh=3.5,
    output_file="../doc/source/charts/average_ft_scores_3h",
    file_formats=['png'],
    colors=None,
    linestyles=None,
    markers=['o', 's'],
    score_name="Spatial Root Mean Square Error",
    min_nb_sta=10,
    availability_ratio=0.75,
    annotation=None,
    xlabel=None,
    nb_of_minor_ticks=(1, 2),
    # output_csv='score.csv',
    averaging='mean',
    score_type='spatial',
)

print("plot_time_scores")
evt.plotting.plot_time_scores(
    list(objs.values()),
    score=sc,
    term=3,
    # hourly_timeseries=True,
    title="Scores at a choosen forecast term\n",
    file_formats=['png'],
    xticking='daily',
    black_axes=True,
    markers=['o', 's'],
    linestyles=['-', '--'],
    colors=['g', 'y'],
    labels=['Ensemble', 'Mocage'],
    # start_end=(dt.date(2017, 6, 2), dt.date(2017, 6, 4)),
    score_name={'RMSE': 'Root mean square error'},
    nb_of_minor_ticks=(6, 1),
    output_file="../doc/source/charts/time_scores_3h",
)

print("station_scores")
obj = list(objs.values())[0]
evt.plotting.plot_station_scores(
    obj, 'RMSE', forecast_day=0,
    title=r"Root mean square error ($\mu g.m^{-3}$)", bbox=[0., 25, 45, 55],
    file_formats=['png'], point_size=20, higher_above=True,
    output_file="../doc/source/charts/station_scores_interp_3h",
    interp2d=True,
    bbox_inches='tight',
)
evt.plotting.plot_station_scores(
    obj, 'RMSE', forecast_day=0,
    title=r"Root mean square error ($\mu g.m^{-3}$)", bbox=[0., 25, 45, 55],
    file_formats=['png'], point_size=20, higher_above=True,
    land_color='#F0EEDC', sea_color='#8AB7DD', cmaplabel=r'RMSE $\mu g/m^3$',
    mark_by=('area', {'urb': 's', 'rur': 'o', 'sub': '^'}),
    output_file="../doc/source/charts/station_scores_3h",
    bbox_inches='tight',
)
evt.plotting.plot_station_scores(
    objs['MFM'], 'RMSE', ref=objs['ENS'], forecast_day=0,
    title=r"Root mean square error difference ($\mu g.m^{-3}$)",
    bbox=[0., 25, 45, 55],
    file_formats=['png'], point_size=20, higher_above=True,
    land_color='#F0EEDC', sea_color='#8AB7DD',
    cmaplabel=r'RMSE difference $\mu g/m^3$',
    mark_by=('area', {'urb': 's', 'rur': 'o', 'sub': '^'}),
    output_file="../doc/source/charts/station_scores_diff_3h",
    bbox_inches='tight',
    cmap='Spectral',
)

print("taylor_diagram")
evt.plotting.plot_taylor_diagram(
    list(objs.values()), colors=['#CB025A', '#004D8A'], markers=['v', '^'],
    labels=objs.keys(), title="Taylor diagram", file_formats=['png'],
    output_file="../doc/source/charts/taylor_diagram_3h",
)

print("score_quartiles")
evt.plotting.plot_score_quartiles(
    list(objs.values()), xscore='FGE', yscore='SpearmanR',
    colors=['#CB025A', '#004D8A'], forecast_day=0, labels=None,
    output_csv=None, invert_xaxis=True, invert_yaxis=False, xmin=None,
    xmax=1, ymin=0, ymax=None, file_formats=['png'], black_axes=True,
    output_file="../doc/source/charts/score_quartiles_3h",
)

print("comparison_scatter_plot")
evt.plotting.plot_comparison_scatter_plot(
    score='RMSE', forecast_day=0, xobject=list(objs.values())[0],
    yobject=list(objs.values())[1], title="Comparison of two simulations",
    file_formats=['png'], black_axes=True,
    output_file="../doc/source/charts/comparison_scatter_plot_3h",
    output_csv=None,
)

print("significant_differences")
persistence = observations.persistence_model()
evt.plotting.plot_significant_differences(
    score_list=['RMSE', 'FGE', 'SpearmanR', 'Bias'],
    former_objects=[list(objs.values())[0],
                    list(objs.values())[0],
                    list(objs.values())[1]],
    later_objects=[list(objs.values())[1], persistence, persistence],
    forecast_day=0, title="", xlabels=None,
    ylabels=["ENS/MFM", "ENS/pers.", "MFM/pers."],
    availability_ratio=0.75, file_formats=['png'],
    output_file="../doc/source/charts/significant_differences_3h",
)

print("diurnal_cycle")
evt.plotting.plot_diurnal_cycle(
    list(objs.values()), station_list=stations.index[:3],
    title="Diurnal Cycle\n\n", file_formats=['png'], black_axes=True,
    output_file="../doc/source/charts/diurnal_cycle_3h",
)
evt.plotting.plot_diurnal_cycle(
    list(objs.values()), station_list=stations.index[:3],
    title="Diurnal Cycle\n\n", file_formats=['png'], plot_type='median',
    envelope=True, black_axes=True,
    output_file="../doc/source/charts/diurnal_cycle_quartiles_3h",
)

print("score_density")
evt.plotting.plot_score_density(
    list(objs.values()),
    'RMSE',
    forecast_day=0,
    availability_ratio=0.75,
    labels=None,
    colors=None,
    linestyles=None,
    # title="plot_score_density",
    # nb_stations=True,
    # annotation='annotation',
    output_file="../doc/source/charts/score_density_3h",
    # file_formats=['png'],
)

print("score_box")
evt.plotting.plot_score_box(
    list(objs.values()),
    'RMSE',
    forecast_day=0,
    availability_ratio=0.75,
    labels=None,
    colors=None,
    # title="plot_score_box",
    nb_stations=True,
    # annotation='annotation',
    output_file="../doc/source/charts/score_box_3h",
    file_formats=['png'],
)

print("data_density")
evt.plotting.plot_data_density(
    list(objs.values()),
    forecast_day=0,
    labels=None,
    colors=None,
    linestyles=['--', '--'],
    # title="plot_data_density",
    file_formats=['png'],
    # xmin=10,
    # xmax=20,
    obs_style={'color': 'k', 'alpha': 0.5, 'label': 'EEA'},
    # annotation='annotation',
    # black_axes=True,
    output_file="../doc/source/charts/data_density_3h",
)

print("data_box")
evt.plotting.plot_data_box(
    list(objs.values()),
    forecast_day=0,
    labels=None,
    colors=None,
    obs_style={'color': 'k', 'alpha': 0.5, 'label': 'EEA'},
    # title="data_box",
    # annotation='annotation',
    file_formats=['png'],
    output_file="../doc/source/charts/data_box_3h",
)

print("target")
with plt.rc_context({'figure.figsize': (11, 5)}):
    evt.fairmode.plot_target_diagram(
        objs['ENS'],
        file_formats=['png'],
        output_file="../doc/source/charts/target_3h",
    )

print("forecast target")
evt.fairmode.plot_forecast_target_diagram(
    objs['ENS'],
    file_formats=['png'],
    output_file="../doc/source/charts/forecast_target_3h",
)

print("summary")
evt.fairmode.plot_fairmode_summary(
    objs['ENS'],
    availability_ratio=0.75,
    forecast_day=0,
    output_file="../doc/source/charts/summary_3h",
    file_formats=['png'],
)

print("tables.median_station_scores")
evt.tables.median_station_scores(
    list(objs.values()), forecast_day=0,
    score_list=['RMSE', 'MeanBias', 'PearsonR'], output_file=None, labels=None,
    availability_ratio=0.75, min_nb_sta=10, float_format='%.2g')

print("median_spatial_scores")
evt.tables.average_scores(
    list(objs.values()), forecast_day=0,
    score_list=['RMSE', 'MeanBias', 'PearsonR'], output_file=None, labels=None,
    availability_ratio=0.75, min_nb_sta=10, float_format='%.2g',
    score_type='spatial',
    averaging='median',
)

print("tables.exceedances_scores")
evt.tables.exceedances_scores(
    daily_objs.values(), forecast_day=0, thresholds=[20, 30, 40],
    output_file=None, labels=None)

print("bar_scores")
evt.plotting.plot_bar_scores(
    objects=list(objs.values()),
    score='RMSE',
    forecast_day=0,
    averaging='median',
    title="Averaged station scores barplot",
    labels=None,
    colors=None,
    subregions=[['AT'], 'all'],
    xtick_labels=['Austria', 'all'],
    output_file="../doc/source/charts/bar_scores_3h",
    file_formats=['png'],
    availability_ratio=0.75,
)

print("bar_exceedances")
evt.plotting.plot_bar_exceedances(
    obj,
    threshold=20.,
    data="obs",
    start_end=None,
    forecast_day=0,
    labels=['Austria', 'others'],
    output_file="../doc/source/charts/bar_exceedances_3h",
    title="Daily threshold exceedances",
    ylabel="Number of exceedances",
    subregions=[['AT'], ['AD', 'CH', 'CZ']],
    bar_kwargs={
        'color': ['#E20B29', '#008C3C'],
        'width': .9,
    },
)

print("bar_contingency_table")
evt.plotting.plot_bar_contingency_table(
    list(objs.values()),
    threshold=20.,
    forecast_day=0,
    start_end=None,
    title="",
    output_file="../doc/source/charts/bar_contingency_table_3h",
)

print("line_exceedances")
evt.plotting.plot_line_exceedances(
    list(daily_objs.values()),
    threshold=20,
    start_end=None,
    forecast_day=0,
    labels=None,
    colors=None,
    linestyles=None,
    markers=None,
    output_file="../doc/source/charts/line_exceedances_3h",
    title="",
    ylabel="Number of incidences",
    xticking='daily',
    date_format='%Y-%m-%d',
    ymin=None,
    ymax=None,
    obs_style={'color': 'g', 'alpha': 0.5, 'label': 'EEA'},
    output_csv=None
)

print("values_scatter_plot")
evt.plotting.plot_values_scatter_plot(
    obj,
    station_list=None,
    start_end=None,
    forecast_day=0,
    title="",
    xlabel='observations',
    ylabel='simulations',
    output_file="../doc/source/charts/values_scatter_plot_3h",
)

print("performance_diagram")
evt.plotting.plot_performance_diagram(
    objs.values(),
    threshold=20,
    output_file="../doc/source/charts/performance_diagram_3h",
)

print("roc_curve")
evt.plotting.plot_roc_curve(
    objects=objs.values(),
    thresholds=[2, 5, 10, 20, 30],
    output_file="../doc/source/charts/roc_curve_3h",
)

print("exceedances_scores")
evt.plotting.plot_exceedances_scores(
    objects=list(daily_objs.values()),
    threshold=20,
    forecast_day=0,
    title="Exceedances scores - threshold = 20",
    labels=['Ensemble', 'Mocage'],
    file_formats=['png'],
    output_file="../doc/source/charts/exceedances_scores_3h",
    output_csv=None,
)

print("summary_bar_chart")
evt.plotting.plot_summary_bar_chart(
    [list(objs.values()), list(objs.values()), list(objs.values())],
    forecast_day=0,
    averaging='mean',
    mean_obs=True,
    title="Summary bar chart",
    labels_list=None,
    colors_list=None,
    groups_labels=None,
    output_file="../doc/source/charts/summary_bar_chart_3h",
    file_formats=['png'],
    availability_ratio=.75,
    adapt_size=0.5,
    ncol=2,
    subplots_adjust={'top': 0.85},
)

print("bar_scores_conc")
evt.plotting.plot_bar_scores_conc(
    objects=list(objs.values()),
    score=sc,
    conc_range=[25, 45, np.inf],
    forecast_day=0,
    averaging='mean',
    title=f"{sc} barplot",
    labels=None,
    colors=None,
    xtick_labels=None,
    output_file="../doc/source/charts/bar_scores_conc_3h",
    file_formats=['png'],
    output_csv=None,
    min_nb_val=2,
    based_on='obs',
    bar_kwargs={},
    annotation=None,
)
