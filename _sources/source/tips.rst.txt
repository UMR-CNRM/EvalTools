Tips
====

Saving Evaluator objects
------------------------

.. highlight:: python

Evaltools includes a convenient way to save
:class:`Evaluator<evaltools.evaluator.Evaluator>` objects
for later use. Let us create such an object as seen in tutorial:

.. ipython:: python

    import evaltools as evt
    from datetime import date
    # import stations with module utils
    stations = evt.utils.read_listing("./sample_data/listing")
    start_date = date(2017, 6, 1)
    end_date = date(2017, 6, 6)
    # create an object of class Observations with module evaluator
    obs = evt.Observations.from_time_series(
        generic_file_path="./sample_data/observations/{year}_co_{station}",
        correc_unit=1e9,
        species='co',
        start=start_date,
        end=end_date,
        stations = stations,
        forecast_horizon=2,
    )
    # create an object of class Simulations with module evaluator
    sim = evt.Simulations.from_time_series(
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
    # create an object of class Evaluator with module evaluator
    obj = evt.Evaluator(obs, sim)
    obj.summary()

Once we have an :class:`Evaluator<evaltools.evaluator.Evaluator>` object, it
is possible to save it using
:func:`evaluator.Evaluator.dump<evaltools.evaluator.Evaluator.dump>` method.

.. ipython:: python

    obj.dump('./sample_data/evaluatorObj.dump')

Once the file is created, it can be loaded anytime with
:func:`evaluator.load<evaltools.evaluator.load>` function.

.. ipython:: python

    obj2 = evt.load('./sample_data/evaluatorObj.dump')
    obj2.summary()
    # objects do not have the same adress, they are considered different
    obj == obj2     
    # but attributes and data have the same values
    obj.stations.equals(obj2.stations)
    obj.obs_df.equals(obj2.obs_df)
    obj.sim_df[0].equals(obj2.sim_df[0])
    obj.sim_df[1].equals(obj2.sim_df[1])


Objects attributes
------------------

:class:`Evaluator<evaltools.evaluator.Evaluator>` class lies on
:class:`Observations<evaltools.evaluator.Observations>`
and :class:`Simulations<evaltools.evaluator.Simulations>` classes. Both of
them use :class:`Dataset<evaltools.dataset.Dataset>` class, which mostly
lies on Dataframes. Let us have an overview of the attributes of these classes: 

:class:`Observations<evaltools.evaluator.Observations>` objects attributes:

.. ipython:: python

    obs.species
    obs.start_date
    obs.end_date
    obs.forecast_horizon
    obs.series_type
    obs.stations
    obs.dataset

:class:`Simulations<evaltools.evaluator.Simulations>` objects attributes:

.. ipython:: python

    sim.species
    sim.start_date
    sim.end_date
    sim.forecast_horizon
    sim.series_type
    sim.stations
    sim.model
    sim.datasets

.. note:: ``sim.datasets`` is a list of
    :class:`Dataset<evaltools.dataset.Dataset>` objects, one for each
    forecast day.

:class:`Dataset<evaltools.dataset.Dataset>` objects attributes:

.. ipython:: python
    
    dt = obs.dataset
    dt.species
    dt.start_date
    dt.end_date
    dt.nb_days
    dt.series_type
    dt.date_format
    type(dt.data)
    dt.data

:class:`Evaluator<evaltools.evaluator.Evaluator>` objects attributes:

.. ipython:: python
    
    obj.species
    obj.start_date
    obj.end_date
    obj.forecast_horizon
    obj.series_type
    obj.model
    obj.stations
    type(obj.obs_df)
    obj.obs_df
    type(obj.sim_df)
    type(obj.sim_df[0])
    obj.sim_df

.. note:: ``obj.obs_df`` is equivalent to ``obj.observations.dataset.data``, and ``obj.sim_df[fd]``
    is equivalent to ``obj.simulations.datasets[fd].data`` (where fd is one of the forecast days).


How to handle data with a time step different from 1h
-----------------------------------------------------

Since version 1.0.4, you can work with data at 1h, 2h, 3h, 4h, 6h and 12h time
step.
The following methods

- :func:`Observations.from_time_series<evaltools.evaluator.Observations.from_time_series>`
- :func:`Observations.from_dataset<evaltools.evaluator.Observations.from_dataset>`
- :func:`Observations.from_nc<evaltools.evaluator.Observations.from_nc>`
- :func:`Simulations.from_time_series<evaltools.evaluator.Simulations.from_time_series>`
- :func:`Simulations.from_dataset<evaltools.evaluator.Simulations.from_dataset>`
- :func:`Simulations.from_nc<evaltools.evaluator.Simulations.from_nc>`

have an argument ``step`` that corresponds to the time step in hours. This
argument is ignored when argument *series_type* is 'daily'.


Plotting with translated annotations
------------------------------------

If you want the annotations on your charts to be translated into French,
you can set ``evaltools.plotting.lang = 'FR'`` in your script.
