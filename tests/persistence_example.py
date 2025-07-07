"""Example of use of the persistence model."""
import evaltools as evt
import datetime as dt

obs_path = "../doc/sample_data/observations/{year}_{species}_{station}"
sim_path = (
    "../doc/sample_data/{model}{runtype}/J{forecastDay}/"
    "{year}_{species}_{station}"
)

listing_path = "../doc/sample_data/listing"
stations = evt.utils.read_listing(
    listing_path, classes='all', types='all',  area_coord=None,
)

start_date = dt.date(2017, 6, 1)
end_date = dt.date(2017, 6, 4)

model_list = ["ENS", "MFM"]
model_colors = {"ENS": '#CB025A', "MFM": '#004D8A'}
species = 'no2'

observations = evt.evaluator.Observations.from_time_series(
    generic_file_path=obs_path.format(
        year="{year}", species=species, station="{station}",
    ),
    correc_unit=1e9,
    species=species,
    start=start_date,
    end=end_date,
    stations=stations,
    forecast_horizon=4,
)

persistence = observations.persistence_model()

observations = observations.sub_period(start_date+dt.timedelta(1), end_date)

objs = {}
daily_objs = {}
fh = 4
for model in model_list:
    generic_file_path = sim_path.format(
        model=model, runtype='forecast', forecastDay='{forecastDay}',
        year='{year}', species=species, station='{station}')
    stations_idx = observations.stations.index
    simulations = evt.evaluator.Simulations.from_time_series(
        generic_file_path=generic_file_path,
        stations_idx=stations_idx,
        forecast_horizon=fh,
        correc_unit=1,
        species=species,
        model=model,
        start=start_date+dt.timedelta(1),
        end=end_date,
    )
    objs[model] = evt.evaluator.Evaluator(
        observations, simulations, color=model_colors[model],
    )
    daily_objs[model] = objs[model].daily_max()

print("plot_median_station_scores")
evt.plotting.plot_median_station_scores(
    list(objs.values())+[persistence],
    score='RMSE',
    title="Median station scores as a function of the forecast term\n",
    file_formats=['png'],
)
