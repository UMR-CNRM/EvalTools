Persistence for model assessment
================================


.. highlight:: python

Persistence model (simulations are equal to the observations of the day
before) may be of interest for model assessment.
From an :class:`Observations<evaltools.evaluator.Observations>`
object you can built an :class:`Evaluator<evaltools.evaluator.Evaluator>`
object corresponding to model persistence with
:meth:`persistence_model<evaltools.evaluator.Observations.persistence_model>`
method.

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
        stations=stations,
        forecast_horizon=2)
    persistence = observations.persistence_model()

.. important:: Remember that the starting day of the new object ``persistence`` is the day after the starting day of ``observations``.

.. ipython:: python

    persistence.summary()

Then, to define other :class:`Evaluator<evaltools.evaluator.Evaluator>`
objects on a period corresponding to the period
of ``persistence``, you can for instance use
:meth:`sub_period<evaltools.evaluator.Observations.sub_period>` method.

.. ipython:: python

    observations = observations.sub_period(start_date+timedelta(days=1), end_date)
    simulations = evt.Simulations.from_time_series(
        generic_file_path=(
            "./sample_data/MFMforecast/J{forecastDay}/{year}_no2_{station}"
        ),
        stations_idx=stations.index,
        species='no2',
        model='MFM',
        start=start_date+timedelta(days=1),
        end=end_date,
        forecast_horizon=2)
    eval_object = evt.Evaluator(
        observations, simulations, color=['#00FFFF'],
    )

And you can now draw the persistence performances on any chart.

.. ipython:: python

    evt.plotting.plot_taylor_diagram([eval_object, persistence],
        output_file="./source/charts/taylor_with_persistence")

.. image:: charts/taylor_with_persistence.png
