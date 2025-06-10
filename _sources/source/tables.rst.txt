Tables module
=============

.. highlight:: python

The :mod:`tables<evaltools.tables>` module gathers functions designed to show
scores computed from :class:`Evaluator<evaltools.evaluator.Evaluator>` objects
in a tabular layout or store them in text files.

Median of station scores
------------------------

The function :func:`median_station_scores<evaltools.tables.median_station_scores>`
displays a given list of scores for a given list of
:class:`Evaluator<evaltools.evaluator.Evaluator>` objects.

Scores from exceedances contingency tables
------------------------------------------

The function :func:`exceedances_scores<evaltools.tables.exceedances_scores>`
displays scores from the contingency table of a given list of
:class:`Evaluator<evaltools.evaluator.Evaluator>` objects.

Example
-------

.. ipython:: python

    import evaltools as evt
    from datetime import date, timedelta
    stations = evt.utils.read_listing("./sample_data/listing")
    start_date = date(2017, 6, 1)
    end_date = date(2017, 6, 4)
    observations = evt.Observations.from_time_series(
        generic_file_path="./sample_data/observations/{year}_no2_{station}",
        correc_unit=1e9,
        species='no2',
        start=start_date,
        end=end_date,
        stations = stations,
        forecast_horizon=2)
    objs = {}
    daily_objs = {}
    for model in ['ENS', 'MFM']:
        generic_file_path = (
            f"../doc/sample_data/{model}forecast/"
            "J{forecast_day}/{year}_no2_{station}"
        )
        stations_idx = observations.stations.index
        simulations = evt.Simulations.from_time_series(
            generic_file_path=generic_file_path,
            stations_idx=stations_idx,
            correc_unit=1,
            species='no2',
            model=model,
            start=start_date,
            end=end_date,
            forecast_horizon=2,
        )
        objs[model] = evt.Evaluator(observations, simulations)
        daily_objs[model] = objs[model].daily_max()

* :func:`median_station_scores<evaltools.tables.median_station_scores>`

.. ipython:: python

    evt.tables.median_station_scores(
        objs.values(),
        forecast_day=0,
        score_list=['RMSE', 'MeanBias', 'PearsonR'],
    )

* :func:`exceedances_scores<evaltools.tables.exceedances_scores>`

.. ipython:: python

    evt.tables.exceedances_scores(
        daily_objs.values(),
        forecast_day=0,
        thresholds=[20, 30, 40],
    )
