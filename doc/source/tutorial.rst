Tutorial
========

Introduction
------------

.. highlight:: python

First you have to import the package.

.. ipython:: python

    import evaltools as evt

In this package, the main class of objects is
:class:`Evaluator<evaltools.evaluator.Evaluator>`.
From an object of this kind you can compute all sort of statistics
and draw charts.
Classes :class:`Observations<evaltools.evaluator.Observations>` and
:class:`Simulations<evaltools.evaluator.Simulations>` are precursors
of :class:`Evaluator<evaltools.evaluator.Evaluator>`. These two classes are
quite similar and the main way to instanciate them is by reading timeseries
files. What we call a timeseries file is a file containing two columns
separated by one or more space, the first column containing times (in
yyyymmddhh format) and the second column containing float values.
Examples of timeseries files can be found in "doc/sample_data".

To begin working with :class:`Evaluator<evaltools.evaluator.Evaluator>`
objects you need a list of stations.
A example of a station listing is located in "doc/sample_data" and can be
read by
:func:`utils.read_listing<evaltools.utils.read_listing>`.

.. ipython:: python

    stations = evt.utils.read_listing("./sample_data/listing")
    stations

We can now instanciate :class:`Observations<evaltools.evaluator.Observations>`
and :class:`Simulations<evaltools.evaluator.Simulations>` objects from
"doc/sample_data" timseries files.

.. ipython:: python

    from datetime import date
    start_date = date(2017, 6, 1)
    end_date = date(2017, 6, 6)
    observations = evt.Observations.from_time_series(
        generic_file_path="./sample_data/observations/{year}_co_{station}",
        correc_unit=1e9,
        species='co',
        start=start_date,
        end=end_date,
        stations = stations,
        forecast_horizon=2,
    )
    simulations = evt.Simulations.from_time_series(
        generic_file_path=(
            "./sample_data/ENSforecast/J{forecastDay}/{year}_co_{station}"
        ),
        stations_idx=stations.index,
        species='co',
        model='ENS',
        start=start_date,
        end=end_date,
        forecast_horizon=2,
    )

To understand the meaning of all the arguments, do not hesitate to refer
to the :doc:`API documentation</source/evaltools>`.

Let's create an :class:`Evaluator<evaltools.evaluator.Evaluator>` object and
start using its methods to compute statistics.

.. ipython:: python

    eval_object = evt.Evaluator(observations, simulations)
    eval_object.temporal_scores(['RMSE', 'FracBias', 'PearsonR'])

Plotting
--------

All plotting functions are gathered in
:mod:`plotting<evaltools.plotting.plotting>`
module. For instance, let's draw mean RMSE over the 2 days forecast period with
:func:`plot_mean_time_scores<evaltools.plotting.plotting.plot_mean_time_scores>`
function.

.. ipython:: python

    evt.plotting.plot_mean_time_scores(
        [eval_object],
        output_file="./source/charts/mean_RMSE_ENS",
        score='RMSE',
    )

And we get:

.. image:: charts/mean_RMSE_ENS.png

If we want more than one simulation drawn on the graph, we just have to
create other :class:`Evaluator<evaltools.evaluator.Evaluator>` objects and
pass them to the plotting function.

.. ipython:: python

    simulations2 = evt.Simulations.from_time_series(
        generic_file_path=(
            "./sample_data/MFMforecast/J{forecastDay}/{year}_co_{station}"
        ),
        stations_idx=stations.index,
        species='co',
        model='MFM',
        start=start_date,
        end=end_date,
        forecast_horizon=2,
    )
    eval_object2 = evt.Evaluator(
        observations, simulations2, color='#00FFFF',
    )
    evt.plotting.plot_mean_time_scores(
        [eval_object, eval_object2],
        output_file="./source/charts/mean_RMSE_MFM",
        score='RMSE',
    )

And we get:

.. image:: charts/mean_RMSE_MFM.png


Different types of series
-------------------------

:class:`Evaluator<evaltools.evaluator.Evaluator>` objects have a series type
attribute

.. ipython:: python

    eval_object.series_type

Here, the series type is ``"hourly"``. Indeed, when we construct an object
from timeseries file, it is the default value which means we work with
data measured at hourly time steps.

Some :class:`Evaluator<evaltools.evaluator.Evaluator>` methods will return an
object with ``seriesType`` attribute equal to ``"daily"``.

For instance,

.. ipython:: python

    daily_max_object =  eval_object.daily_max()
    daily_max_object.series_type

We have thus created a new :class:`Evaluator<evaltools.evaluator.Evaluator>`
object which data is now composed of daily maximum values. Let's compare
observation data held within ``eval_object`` and ``daily_max_object`` for a
given station.

.. ipython:: python

    eval_object.obs_df['AT0VOR1']

.. ipython:: python

    daily_max_object.obs_df['AT0VOR1']

Data with ``daily_max_object`` is given at daily time steps. Yet we can still
apply statical methods to this object to get scores per station for instance:

.. ipython:: python

    daily_max_object.temporal_scores(['RMSE', 'FracBias', 'PearsonR'])
