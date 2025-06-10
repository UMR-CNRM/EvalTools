# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module defines the classes managing the data."""

import numpy as np
import pandas as pd
from datetime import timedelta
from os.path import isfile
import pickle
import copy
from functools import reduce
import re
import warnings

import evaltools as evt
from evaltools._deprecate import deprecate
from evaltools._deprecate import deprecate_kwarg
from evaltools._deprecate import deprecate_attrs


@deprecate_attrs(
    start_date='startDate',
    end_date='endDate',
    series_type='seriesType',
    forecast_horizon='forecastHorizon',
)
class Observations(object):
    """
    Class gathering observations of the studied case.

    An object of class Observations will be specific to one species,
    one series type and one period.

    """

    @deprecate_kwarg('startDate', 'start_date')
    @deprecate_kwarg('endDate', 'end_date', stacklevel=3)
    @deprecate_kwarg('seriesType', 'series_type', stacklevel=4)
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=5)
    def __init__(self, species, start_date, end_date, stations, series_type,
                 forecast_horizon=1, step=1, path=''):
        """
        Observations constructor.

        Parameters
        ----------
        species : str
            Species name (ex: "o3").
        start_date : datetime.date
            Starting day of the studied period.
        end_date : datetime.date
            Ending day (included) of the studied period.
        stations : pandas.DataFrame
            DataFrame with station names as index, and metadata variables
            as columns.
        series_type : str.
            It can be 'hourly' or 'daily'.
        forecast_horizon : int
            Number of day corresponding to the forecast horizon of the model.
            For example, if the forcast horizon is 4 days, the end date of the
            studied observations has to be 3 days further than the end of the
            studied period since the studied period corresponds to the period
            along which the model has been executed.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        """
        self.stations = stations
        self.forecast_horizon = forecast_horizon
        self.path = path
        # main attribute
        self.dataset = evt.dataset.Dataset(
            stations=stations.index,
            start_date=start_date,
            end_date=end_date+timedelta(days=forecast_horizon-1),
            species=species,
            series_type=series_type,
            step=step)

    @property
    def start_date(self):
        """Get the starting date of the object."""
        return self.dataset.start_date

    @property
    def end_date(self):
        """Get the ending date of the object."""
        return self.dataset.end_date - timedelta(days=self.forecast_horizon-1)

    @property
    def species(self):
        """Get the species of the object."""
        return self.dataset.species

    @property
    def series_type(self):
        """Get the series type of the object."""
        return self.dataset.series_type

    @property
    def step(self):
        """Get the time step of the object."""
        return self.dataset.step

    @property
    def freq(self):
        """Get the time step of the object as a datetime.timedelta."""
        return self.dataset.freq

    @property
    def obs_df(self):
        """Get the observation data of the object."""
        return self.dataset.data

    @classmethod
    @deprecate_kwarg('seriesType', 'series_type')
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=3)
    def from_time_series(
            cls, generic_file_path, species, start, end, stations,
            correc_unit=1, series_type='hourly', forecast_horizon=1,
            availability_ratio=False, step=1):
        """
        Class method used to construct an object from timeseries files.

        Parameters
        ----------
        generic_file_path : str
            Generic path of timeseriesfiles with {year} instead of
            the year number and {station} instead of the station name.
        species : str
            Species name (ex: "o3").
        start : datetime.date
            Starting day of the studied period.
        end : datetime.date
            Ending day (included) of the studied period.
        stations : pandas.DataFrame
            DataFrame with station names as index, and metadata variables
            as columns.
        correc_unit : float
            multiplicative factor applied to original values.
        series_type : str
            Must be equal to 'hourly' if time series files contain one
            row per hour, or equal to 'daily' if they contain one row
            per day.
        forecast_horizon : int
            Number of day corresponding to the forecast horizon of the model.
            For example, if the forcast horizon is 4 days, the end date of
            the studied observations has to be 3 days further than the end
            of the studied period since the studied period corresponds to
            the period along which the model has been executed.
        availability_ratio : float or None
            Minimal rate of data available on the period required to
            keep a station. If None, stations with only nan values
            are dropped.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            evaltools.evaluator.Observations object

        """
        obj = cls(series_type=series_type, species=species,
                  start_date=start, end_date=end, stations=stations,
                  forecast_horizon=forecast_horizon, step=step,
                  path=generic_file_path)
        obj.dataset.update_from_time_series(generic_file_path, correc_unit)
        if availability_ratio is not False:
            obj.drop_unrepresentative_stations(availability_ratio)
        return obj

    @classmethod
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon')
    def from_dataset(
            cls, ds, forecast_horizon=1, correc_unit=1,
            listing_path=None, step=1, **kwargs):
        """
        Initialize from an evaltools.dataset.Dataset object.

        Parameters
        ----------
        ds : evaltools.dataset.Dataset object
            Dataset object corresponding observation data.
        forecast_horizon : int
            Number of forecasted days.
        correc_unit : float
            Factor to apply to original values.
        listing_path : str
            Path of the station listing where to retrieve metadata variables.
            This listing is optional and only used to get metadata.
        step : int
            Time step in hours (ignored if series_type == 'daily').
        **kwargs :
            These parameters (like 'sep', 'sub_list', ...) will be
            passed to evaltools.utils.read_listing().

        Returns
        -------
            evaltools.evaluator.Observations object

        """
        stations = copy.deepcopy(ds.metadata)

        if listing_path is not None:
            listing_data = evt.utils.read_listing(listing_path, **kwargs)
            if not np.in1d(ds.data.columns, listing_data.index).all():
                print("Warning: listing does not contain all stations of "
                      "the Dataset. Missing stations are:")
                missing_stations = ds.data.columns[
                    ~np.in1d(ds.data.columns, listing_data.index)
                ].tolist()
                missing_stations.sort()
                for sta in missing_stations:
                    print(sta)
            stations = stations.merge(
                listing_data,
                how='left',
                left_index=True,
                right_index=True,
                suffixes=('', '_listing')).drop_duplicates()

        obj = cls(species=ds.species, start_date=ds.start_date,
                  end_date=ds.end_date-timedelta(days=forecast_horizon-1),
                  stations=stations, series_type=ds.series_type,
                  forecast_horizon=forecast_horizon, step=step)
        obj.dataset.update_from_dataset(ds, correc_unit)
        obj.dataset.data.where(obj.dataset.data >= 0, np.nan, inplace=True)
        return obj

    @classmethod
    @deprecate_kwarg('seriesType', 'series_type')
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=3)
    def from_nc(cls, start, end, paths, species, series_type='hourly',
                forecast_horizon=1, group=None, dim_names={},
                coord_var_names={}, listing_path=None, metadata_var={},
                correc_unit=1, step=1, **kwargs):
        """
        Construct an object from netcdf files.

        To be handle by this method, netcdf variables must be 2-dimensional:
        the first dimension corresponding to time and the second one to
        the diferent measurement sites.

        Parameters
        ----------
        start : datetime.date
            Starting day of the studied period.
        end : datetime.date
            Ending day (included) of the studied period.
        paths : list of str
            List of netcdf files where to retrieve concentration values.
        species : str
            Species name that must correspond to the name of the retrieved
            netcdf variable.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        forecast_horizon : int
            Number of forecasted days.
        group : None or str
            Group to read within the netcdf file. If equal to None, the root
            group is read.
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        listing_path : str
            Path of the station listing where to retrieve metadata variables.
            This listing is optional and only used to get metadata.
        metadata_var : dict
            Dictionary that define metadata variables to get from de netcdf
            file. Keys of the provided dictionary are variable names as
            found in the file, and its values are variable names used for
            the returned dataset. These metadata variables must have one
            dimension only, corresponding to the station codes.
        correc_unit : float
            Factor to apply to original values.
        step : int
            Time step in hours (ignored if series_type == 'daily').
        **kwargs :
            These parameters (like 'sep', 'sub_list', ...) will be
            passed to evaltools.utils.read_listing().

        Returns
        -------
            evaltools.evaluator.Observations object

        """
        store_list = [evt.dataset.Store(p, group=group, read_only=True,
                                        dim_names=dim_names,
                                        coord_var_names=coord_var_names,
                                        series_type=series_type)
                      for p in paths]
        ds_list = [store.get_dataset(
                        name=species,
                        dataset_name=species,
                        start_date=start,
                        end_date=end+timedelta(
                            days=forecast_horizon-1),
                        metadata_var=metadata_var,
                        series_type=series_type,
                        step=step,
                        )
                   for store in store_list]

        if len(ds_list) > 1:
            obs_ds = evt.dataset.Dataset(
                stations=reduce(np.union1d, [d.data.columns for d in ds_list]),
                start_date=start,
                end_date=end+timedelta(days=forecast_horizon-1),
                species=species,
                series_type=series_type,
                step=step)
            for ds in ds_list:
                obs_ds.update_from_dataset(ds)
            obs_ds._metadata = pd.concat([d._metadata for d in ds_list])
            obs_ds._metadata = obs_ds._metadata.loc[
                ~obs_ds._metadata.index.duplicated()]
        else:
            obs_ds = ds_list[0]

        observations = cls.from_dataset(
            obs_ds,
            listing_path=listing_path,
            forecast_horizon=forecast_horizon,
            step=step,
            correc_unit=correc_unit,
            **kwargs)

        return observations

    @classmethod
    @deprecate_kwarg('seriesType', 'series_type')
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=3)
    def from_sqlite(
            cls, start, end, paths, species, table, time_key_name='dt',
            series_type='hourly', step=1, forecast_horizon=1,
            listing_path=None, correc_unit=1, **kwargs):
        """
        Construct an object from netcdf files.

        To be handle by this method, netcdf variables must be 2-dimensional:
        the first dimension corresponding to time and the second one to
        the diferent measurement sites.

        Parameters
        ----------
        start : datetime.date
            Starting day of the studied period.
        end : datetime.date
            Ending day (included) of the studied period.
        paths : list of str
            List of netcdf files where to retrieve concentration values.
        species : str
            Species name.
        table : str
            Name of the table to read in the sqlite file.
        time_key_name : str
            Unique id of the sqlite table corresponding to the time of the
            observations.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        step : int
            Time step in hours (ignored if series_type == 'daily').
        forecast_horizon : int
            Number of forecasted days.
        listing_path : str
            Path of the station listing where to retrieve metadata variables.
            This listing is optional and only used to get metadata.
        correc_unit : float
            Factor to apply to original values.
        **kwargs :
            These parameters (like 'sep', 'sub_list', ...) will be
            passed to evaltools.utils.read_listing().

        Returns
        -------
            evaltools.evaluator.Observations object

        """
        store_list = [
            evt.sqlite.Store(
                p, table=table, time_key_name=time_key_name,
                create_if_not_exist=False,
            )
            for p in paths
        ]
        ds_list = [
            store.get_dataset(
                dataset_name=species,
                start_date=start,
                end_date=end+timedelta(days=forecast_horizon-1),
                series_type=series_type,
                step=step,
            )
            for store in store_list
        ]

        if len(ds_list) > 1:
            obs_ds = evt.dataset.Dataset(
                stations=reduce(np.union1d, [d.data.columns for d in ds_list]),
                start_date=start,
                end_date=end+timedelta(days=forecast_horizon-1),
                species=species,
                series_type=series_type,
                step=step,
            )
            for ds in ds_list:
                obs_ds.update_from_dataset(ds)
            obs_ds._metadata = pd.concat([d._metadata for d in ds_list])
            obs_ds._metadata = obs_ds._metadata.loc[
                ~obs_ds._metadata.index.duplicated()]
        else:
            obs_ds = ds_list[0]

        observations = cls.from_dataset(
            obs_ds,
            listing_path=listing_path,
            forecast_horizon=forecast_horizon,
            step=step,
            correc_unit=correc_unit,
            **kwargs,
        )

        return observations

    def drop_unrepresentative_stations(self, availability_ratio=0.75):
        """
        Drop stations with a certain rate of missing values.

        Modify Dataset object in place.

        Parameters
        ----------
        availability_ratio : float or None
            Minimal rate of data available on the period required to
            keep a station. If None, stations with only nan values
            are dropped.

        """
        dropped_sta = self.dataset.drop_unrepresentative_stations(
            availability_ratio
        )
        print(
            "{} stations dropped over {}.".format(
                dropped_sta.shape[0], self.stations.shape[0]
            )
        )
        self.stations = self.stations.loc[self.dataset.data.columns]

    @deprecate_kwarg('filePath', 'file_path')
    def to_csv(self, file_path):
        """
        Save timeseries dataframe as csv.

        Parameters
        ----------
        file_path : str
            Csv file path.

        """
        self.obs_df.to_csv(
            file_path,
            sep=',',
            na_rep='nan',
            float_format='%g',
            header=True,
            index=True,
            date_format=self.dataset.date_format,
        )

    @deprecate_kwarg('filePath', 'file_path')
    def check_values(self, threshold, drop=False, file_path=None):
        """
        Check if observation values exceed a threshold.

        If there are values above the threshold, a message is printed
        and these values are set to nan if drop == True.

        Parameters
        ----------
        threshold : scalar
            Threshold value.
        drop : bool
            If True, values above the threshold are set to nan.
        file_path : None or str
            File path where to save the names of stations that exceed
            the threshold.

        """
        print("Observations threshold exceedances check:")
        self.dataset.check_threshold(threshold, drop, file_path)

    def persistence_model(self, color='k'):
        """
        Build an Evaluator object based on persistence model.

        Parameters
        ----------
        color : str
            Default color that will be used in plotting functions
            for the new object.

        Returns
        -------
            Evaluator object.

        """
        simulations = Simulations(
            start_date=self.start_date+timedelta(days=1),
            end_date=self.end_date,
            stations=self.stations.index,
            species=self.species,
            model="persistence",
            series_type=self.series_type,
            forecast_horizon=self.forecast_horizon,
            step=self.step)
        for fday in range(self.forecast_horizon):
            df = pd.DataFrame(
                self.dataset.data.values,
                columns=self.dataset.data.columns,
                index=self.dataset.data.index+timedelta(days=fday+1))
            simulations.datasets[fday].update_from_dataset(df)
        return Evaluator(
            self.sub_period(self.start_date+timedelta(days=1), self.end_date),
            simulations,
            color)

    @deprecate_kwarg('startDate', 'start_date')
    @deprecate_kwarg('endDate', 'end_date', stacklevel=3)
    def sub_period(self, start_date, end_date):
        """
        Build a new Observations object define on a shorter period.

        Parameters
        ----------
        start_date : datetime.date
            Starting date of the new object.
        end_date : datetime.date
            Ending date of the new object.

        Returns
        -------
            Observations object.

        """
        # check args integrity
        if start_date < self.start_date:
            raise evt.EvaltoolsError(
                "start_date must be >= {} !!!".format(self.start_date))
        if end_date > self.end_date:
            raise evt.EvaltoolsError(
                "end_date must be <= {} !!!".format(self.end_date))

        new_obs = self.__class__(
            species=self.species, start_date=start_date, end_date=end_date,
            stations=self.stations, series_type=self.series_type,
            forecast_horizon=self.forecast_horizon, step=self.step)
        idx = np.logical_and(
            self.dataset.data.index.date >= start_date,
            self.dataset.data.index.date <= end_date+timedelta(
                days=self.forecast_horizon-1))
        new_obs.dataset.data = self.dataset.data.loc[idx]

        return new_obs

    @deprecate_kwarg('outputFile', 'output_file')
    def sim_vs_obs(
            self, grid, time, point_size=20, vmin=None, vmax=None,
            cmap=None, colors=None, bounds=None, output_file=None,
            file_formats=['png']):
        """
        Scatter plot of observations above simulation raster.

        Parameters
        ----------
        grid : evaltools.interpolation.Grid
            Grid object that must contain data for the species of
            the current Observations object.
        time : datetime.datetime or datetime.date
            Time for which to plot the observation values. This time
            must be contained in the current Observations object.
        point_size : float
            Point size (as define in matplotlib.pyplot.scatter).
        vmin, vmax : None or scalar
            Min and max values for the legend colorbar. If None, these
            values are found automatically.
        cmap : None or matplotlib.colors.Colormap object
            Colors used for plotting (default: matplotlib.cm.jet).
        colors : None or list of str
            List of color used in the chart if you want to discretize
            the values.
        bounds : None or list of scalar
            Boundary values for each category if you want to discretize
            the values. Arguments vmin and vmax must not be None, and the
            boundary values contained between vmin and vmax.
            Ignored if colors is None.
        output_file : None str
            File where to save the plot (without extension).
            If None, the figure is shown in a popping window.
        file_formats : list of str
            List of file extensions.

        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.basemap import Basemap
        import matplotlib as mpl

        # define default colormap
        if cmap is None:
            cmap = cm.jet

        # plotting
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.add_subplot(111)

        # draw coast lines with Basemap
        m = Basemap(resolution='i', projection='merc',
                    llcrnrlat=grid.minLat, urcrnrlat=grid.maxLat,
                    llcrnrlon=grid.minLon, urcrnrlon=grid.maxLon,
                    lat_ts=(grid.minLat + grid.maxLat)/2)
        m.drawcoastlines(linewidth=0.5, color='k', zorder=1)

        # discretizing values if colors is not None
        norm = None
        if colors is not None:
            cmap = mpl.colors.ListedColormap(colors)
            if bounds is not None:
                bounds = [vmin] + bounds + [vmax]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # plot simulation raster
        m.imshow(
            grid.grids[self.species],
            extent=(grid.minLon, grid.maxLon, grid.minLat, grid.maxLat),
            zorder=0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm)

        # plot observation values
        m.scatter(
            list(self.stations['lon']),
            list(self.stations['lat']),
            c=self.obs_df.loc[time],
            latlon=True,
            edgecolors='k',
            linewidths=1,
            s=point_size,
            zorder=2,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax
            )
        m.colorbar(ticks=bounds)

        # save figure
        evt.plotting._save_figure(output_file, file_formats)

    def daily_mean(self, availability_ratio=0.75):
        """
        Build Observations object working on daily mean.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily mean.

        Returns
        -------
        evaluator.Observations
            Observations object with series_type = 'daily' and with data
            corresponding to the computed daily mean.

        """
        if self.series_type not in ['hourly']:
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "to use this method !!!"
            )
            return None

        new_obs = self.__class__(
            species=self.species, start_date=self.start_date,
            end_date=self.end_date, stations=self.stations,
            series_type='daily', forecast_horizon=self.forecast_horizon)
        new_obs.dataset.update_from_dataset(
            evt.timeseries.daily_mean(self.dataset.data, availability_ratio))
        return new_obs

    def daily_max(self, availability_ratio=0.75):
        """
        Return Observations object working on daily maximum.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily maximum.

        Returns
        -------
        evaluator.Observations
            Observations object with series_type = 'daily' and with data
            corresponding to the computed daily maximum.

        """
        if self.series_type not in ['hourly']:
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "to use this method !!!"
            )
            return None

        new_obs = self.__class__(
            species=self.species, start_date=self.start_date,
            end_date=self.end_date, stations=self.stations,
            series_type='daily', forecast_horizon=self.forecast_horizon)
        new_obs.dataset.update_from_dataset(
            evt.timeseries.daily_max(self.dataset.data, availability_ratio))
        return new_obs

    def moving_average_daily_max(self, availability_ratio=0.75):
        """
        Compute the daily maximum of the 8-hour moving average.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available to compute the average for a
            given 8-hour window, and also minimal rate of values available
            for a given day to compute the daily maximum of the moving
            average. For example with availability_ratio=0.75, a daily maximum
            eight hours average can only be calculated if 18 eight hours
            average are available each of which requires 6 hourly values to
            be available.

        Returns
        -------
        evaluator.Observations
            Observations object with series_type = 'daily' and with data
            corresponding to the computed daily maximum of the moving average.

        """
        if self.series_type not in ['hourly'] or self.step != 1:
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "with one hour time step to use this method !!!"
            )
            return None

        new_obs = self.__class__(
            species=self.species, start_date=self.start_date,
            end_date=self.end_date, stations=self.stations,
            series_type='daily', forecast_horizon=self.forecast_horizon)

        temp_df = copy.deepcopy(self.dataset.data)
        # add nan for the day before since 1st window is 17 to 1 o'clock
        for new_row in temp_df.index[0:24]-timedelta(days=1):
            temp_df.loc[new_row] = np.nan
        temp_df.sort_index(inplace=True)
        temp_df = evt.timeseries.moving_average_daily_max(
            temp_df, availability_ratio)

        new_obs.dataset.update_from_dataset(temp_df)

        return new_obs

    def filtered_series(self, availability_ratio=0.75):
        """
        Return Observations object working on the filtered series.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available for a given hour to compute the
            daily filtered series for this hour in all days.

        Returns
        -------
        evaluator.Observations
            Observations object with series_type = 'hourly' and with data
            corresponding to the computed filtered series.

        """
        if self.series_type != 'hourly':
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "to use this method !!!"
            )
            return None

        new_obs = self.__class__(
            species=self.species, star_d_date=self.star_d_date,
            en_d_date=self.en_d_date, stations=self.stations,
            series_type='hourly', forecast_horizon=self.forecast_horizon)
        new_obs.dataset.update_from_dataset(
            evt.timeseries.filtered_series(self.dataset.data,
                                           availability_ratio))
        return new_obs

    def normalized_series(self):
        """
        Return Observations object with normalized series.

        This method normalizes each series of observations by substracting
        its median and dividing by its interquartile range.

        Returns
        -------
        evaluator.Observations
            Observations object  with data corresponding to the computed
            normalized series.

        """
        new_obs = self.__class__(
            species=self.species,
            start_date=self.start_date,
            end_date=self.end_date,
            stations=self.stations,
            series_type=self.series_type,
            forecast_horizon=self.forecast_horizon,
        )
        new_obs.dataset.update_from_dataset(
            evt.timeseries.normalized_series(self.dataset.data)
        )
        return new_obs


@deprecate_attrs(
    start_date='startDate',
    end_date='endDate',
    series_type='seriesType',
    sim_df='simDF',
    forecast_horizon='forecastHorizon',
)
class Simulations(object):
    """
    Class gathering simulations of the studied case.

    An object of class Simulations will be specific to one species,
    one series type, one period and one model.

    Parameters
    ----------
    start_date : datetime.date
        Start day of the studied period.
    end_date : datetime.date
        End day (included) of the studied period.
    stations : 1D array-like of str
        List of the names of studied stations.
    species : str
        Species name (ex: "o3").
    model : str
        Name of the model that produced the simulated data.
    series_type : str
        It can be 'hourly' or 'daily'.
    forecast_horizon : int
        Number of day corresponding to the forcast horizon of the model.
    step : int
        Time step in hours (ignored if series_type == 'daily').

    """

    @deprecate_kwarg('startDate', 'start_date')
    @deprecate_kwarg('endDate', 'end_date', stacklevel=3)
    @deprecate_kwarg('seriesType', 'series_type', stacklevel=4)
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=5)
    def __init__(self, start_date, end_date, stations, species, model,
                 series_type, forecast_horizon, step=1, path=''):
        """
        Build a Simulations object.

        Parameters
        ----------
        start_date : datetime.date
            Start day of the studied period.
        end_date : datetime.date
            End day (included) of the studied period.
        stations : 1D array-like of str
            List of the names of studied stations.
        species : str
            Species name (ex: "o3").
        model : str
            Name of the model that produced the simulated data.
        series_type : str
            It can be 'hourly' or 'daily'.
        forecast_horizon : int
            Number of day corresponding to the forcast horizon of the model.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        """
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_horizon = forecast_horizon
        self.model = model
        self.series_type = series_type
        self.path = path
        # main attribute
        self.datasets = []
        for fd in range(forecast_horizon):
            self.datasets.append(
                evt.dataset.Dataset(
                    stations=stations,
                    start_date=start_date+timedelta(days=fd),
                    end_date=end_date+timedelta(days=fd),
                    species=species,
                    series_type=series_type,
                    step=step,
                )
            )

    @property
    def stations(self):
        """Get the station list of the object."""
        return self.datasets[0].data.columns

    @property
    def sim_df(self):
        """Get the simulation data of the object."""
        return [self.datasets[fd].data
                for fd in range(self.forecast_horizon)]

    @property
    def step(self):
        """Get the time step of the object."""
        return self.datasets[0].step

    @property
    def freq(self):
        """Get the time step of the object as a datetime.timedelta."""
        return self.datasets[0].freq

    @classmethod
    @deprecate_kwarg('stationsIdx', 'stations_idx')
    @deprecate_kwarg('seriesType', 'series_type', stacklevel=3)
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=4)
    def from_time_series(
            cls, generic_file_path, stations_idx, species, model,
            start, end, forecast_horizon=1, correc_unit=1,
            series_type='hourly', availability_ratio=False,
            step=1):
        """
        Class method used to construct an object from timeseries files.

        Parameters
        ----------
        generic_file_path : str
            Generic path of timeseriesfiles with {year} instead of
            the year number, {station} instead of the station name
            and {forecastDay} instead of the forecast day number.
        stations_idx : list of str
            List of the names of studied stations.
        species : str
            Species name (ex: "o3").
        model : str
            Name of the model that produced the simulated data.
        start : datetime.date
            Start day of the studied period.
        end : datetime.date
            End day (included) of the studied period.
        forecast_horizon : int
            Number of day corresponding to the forcast horizon of the model.
        correc_unit : float
            Multiplicative factor applied to original values.
        series_type : str
            It can be 'hourly' or 'daily'.
        availability_ratio : float or None
            Minimal rate of data available on the period required to
            keep a station. If None, stations with only nan values
            are dropped.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        """
        obj = cls(start_date=start, end_date=end, stations=stations_idx,
                  species=species, model=model, series_type=series_type,
                  forecast_horizon=forecast_horizon, step=step,
                  path=generic_file_path)
        for fd in range(forecast_horizon):
            obj.datasets[fd].update_from_time_series(
                generic_file_path.format(year="{year}",
                                         station="{station}",
                                         forecastDay=fd,
                                         forecast_day=fd),
                correc_unit)
        if availability_ratio is not False:
            obj.drop_unrepresentative_stations(availability_ratio)
        return obj

    @classmethod
    @deprecate_kwarg('stationsIdx', 'stations_idx')
    def from_dataset(
            cls, model, ds_list, stations_idx=None, correc_unit=1,
            step=1, path=''):
        """
        Initialize from an evaltools.dataset.Dataset object.

        Stations kept in the returned Dataset are the intersection
        of stations from the Dataset objects of ds_list.

        Parameters
        ----------
        model : str
            Name of the model.
        ds_list : list of evaltools.dataset.Dataset objects
            Dataset objects corresponding to the different
            forecast terms. The order in the list matters.
        stations_idx : list of str
            List of the names of studied stations.
        correct_unit : float
            Factor to apply to original values.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            evaltools.evaluator.Simulations object

        """
        # define station list as the intersection of stations of the datasets
        stations = reduce(np.intersect1d,
                          [ds.data.columns for ds in ds_list] +
                          [stations_idx]*(stations_idx is not None))

        # check metadata
        forecast_horizon = len(ds_list)
        species = ds_list[0].species
        start_date = ds_list[0].start_date
        end_date = ds_list[0].end_date
        series_type = ds_list[0].series_type
        for fd, ds in enumerate(ds_list):
            if (species != ds.species or
                    ds.start_date != (start_date+timedelta(days=fd)) or
                    ds.end_date != (end_date+timedelta(days=fd)) or
                    series_type != ds.series_type):
                raise evt.EvaltoolsError(
                    "Metadata of Dataset objects from "
                    "ds_list are not compatible !!!"
                )

        # build Simulations object
        obj = cls(start_date=start_date, end_date=end_date,
                  stations=stations, species=species, model=model,
                  series_type=series_type, forecast_horizon=forecast_horizon,
                  step=step, path=path)
        for fd, ds in enumerate(ds_list):
            obj.datasets[fd].update_from_dataset(ds, correc_unit)
            obj.datasets[fd].data.where(obj.datasets[fd].data >= 0, np.nan,
                                        inplace=True)

        return obj

    @classmethod
    @deprecate_kwarg('seriesType', 'series_type')
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=3)
    def from_nc(cls, model, start, end, paths, species, series_type='hourly',
                stations=None, forecast_horizon=1, group=None, dim_names={},
                coord_var_names={}, correc_unit=1, step=1):
        """
        Construct an object from netcdf files.

        To be handle by this method, netcdf variables must be 2-dimensional:
        the first dimension corresponding to time and the second one to
        the diferent measurement sites.

        Parameters
        ----------
        model : str
            Name of the model.
        start : datetime.date
            Starting day of the studied period.
        end : datetime.date
            Ending day (included) of the studied period.
        species : str
            Species name that must correspond to the name of the retrieved
            netcdf variable.
        forecast_horizon : int
            Forecast term given as the number of forecasted days.
        paths : list or dictionary
            If 'paths' is given as a dictionary, keys must be 'D0', 'D1', ...
            corresponding to the forecast terms, and values must be lists
            of str corresponding to paths to netcdf files where to find
            concentration values.
            If 'paths' is given as a list, this list will be used for every
            forcast terms. Thus, either forecast_horizon=1 or the netcdf group
            to fetch in the files must depends on the forecast term (ex:
            group='D{fd}').
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        stations : None or list of str
            List of the names of studied stations. If None, all stations
            are kept.
        group : None or str
            Group to read within the netcdf file. If equal to None, the root
            group is read. The group name you provide can contains '{fd}'
            that will be replaced according to the number of the forecasted
            day (ex: group='D{fd}').
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        correc_unit : float
            Factor to apply to original values.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            evaltools.evaluator.Simulations object

        """
        sim_ds_list = []
        for fd in range(forecast_horizon):
            # check paths type
            if isinstance(paths, dict):
                try:
                    paths_list = paths['D{}'.format(fd)]
                    assert isinstance(paths_list, list)
                except (KeyError, AssertionError):
                    raise evt.EvaltoolsError(
                        (
                            "If you provided 'paths' argument as a "
                            "dictionary, expected keys are {} and values "
                            "must be lists of str corresponding to "
                            "paths to netcdf files."
                        ).format(
                            [
                                'D{}'.format(fd)
                                for fd in range(forecast_horizon)
                            ]
                        )
                    )
            elif isinstance(paths, list):
                if forecast_horizon > 1:
                    if '{fd}' not in group:
                        raise evt.EvaltoolsError(
                            "As you provide 'paths' argument as a list with "
                            "a forecast horizon greater than 1, the group "
                            "name must be mutable according to the "
                            "forecasted day by containing '{fd}'.")
                paths_list = paths
            else:
                raise evt.EvaltoolsError(
                    "'paths' argument must be either a dictionary or a list."
                )

            # allow group to contain the forecast day number
            if group is not None:
                gp = group.format(fd=fd)
            else:
                gp = group

            # open netcdf files
            store_list = [evt.dataset.Store(p, group=gp, read_only=True,
                                            dim_names=dim_names,
                                            coord_var_names=coord_var_names,
                                            series_type=series_type)
                          for p in paths_list]

            # get concentration values
            ds_list = [store.get_dataset(
                            name=species,
                            dataset_name=species,
                            start_date=start+timedelta(days=fd),
                            end_date=end+timedelta(days=fd),
                            series_type=series_type,
                            step=step,
                            )
                       for store in store_list]

            if len(ds_list) > 1:
                sim_ds = evt.dataset.Dataset(
                    stations=reduce(np.union1d,
                                    [d.data.columns for d in ds_list]),
                    start_date=start+timedelta(days=fd),
                    end_date=end+timedelta(days=fd),
                    species=species,
                    series_type=series_type,
                    step=step,
                )
                for ds in ds_list:
                    sim_ds.update_from_dataset(ds)
            else:
                sim_ds = ds_list[0]
            sim_ds_list.append(sim_ds)

        simulations = cls.from_dataset(
            model=model,
            ds_list=sim_ds_list,
            stations_idx=stations,
            step=step,
            correc_unit=correc_unit,
        )
        return simulations

    @classmethod
    @deprecate_kwarg('seriesType', 'series_type')
    @deprecate_kwarg('forecastHorizon', 'forecast_horizon', stacklevel=3)
    def from_sqlite(
            cls, model, start, end, paths, species, table, time_key_name='dt',
            series_type='hourly', step=1, forecast_horizon=1,
            stations=None, correc_unit=1):
        """
        Construct an object from sqlite files.

        To be handle by this method, sqlite tables must have their unique
        id corresponding to the time and columns corresponding to the
        measurements sites.

        Parameters
        ----------
        model : str
            Name of the model.
        start : datetime.date
            Starting day of the studied period.
        end : datetime.date
            Ending day (included) of the studied period.
        paths : list or dictionary
            If 'paths' is given as a dictionary, keys must be 'D0', 'D1', ...
            corresponding to the forecast terms, and values must be lists
            of str corresponding to paths to sqlite files where to find
            concentration values.
            If 'paths' is given as a list, this list will be used for every
            forcast terms. Thus, either forecast_horizon=1 or the sqlite table
            name must depend on the forecast term (ex: 'D{fd}').
        species : str
            Species name.
        table : str
            Name of the table to read in the sqlite file.
        time_key_name : str
            Unique id of the sqlite table corresponding to the time of the
            observations.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        step : int
            Time step in hours (ignored if series_type == 'daily').
        forecast_horizon : int
            Forecast term given as the number of forecasted days.
        stations : None or list of str
            List of the names of studied stations. If None, all stations
            are kept.
        correc_unit : float
            Factor to apply to original values.

        Returns
        -------
            evaltools.evaluator.Simulations object

        """
        sim_ds_list = []
        for fd in range(forecast_horizon):
            # check paths type
            if isinstance(paths, dict):
                try:
                    paths_list = paths['D{}'.format(fd)]
                    assert isinstance(paths_list, list)
                except (KeyError, AssertionError):
                    raise evt.EvaltoolsError(
                        (
                            "If you provided 'paths' argument as a " +
                            "dictionary, expected keys are {} and values " +
                            "must be lists of str corresponding to paths to " +
                            "netcdf files."
                        ).format(
                            [
                                'D{}'.format(fd)
                                for fd in range(forecast_horizon)
                            ]
                        )
                    )
            elif isinstance(paths, list):
                if forecast_horizon > 1:
                    if '{fd}' not in table:
                        raise evt.EvaltoolsError(
                            "As you provide 'paths' argument as a list with " +
                            "a forecast horizon greater than 1, the group " +
                            "name must be mutable according to the " +
                            "forecasted day by containing '{fd}'.")
                paths_list = paths
            else:
                raise evt.EvaltoolsError(
                    "'paths' argument must be either a dictionary or a list.")

            # open sqlite files
            store_list = [
                evt.sqlite.Store(
                    p, table=table.format(fd=fd), time_key_name=time_key_name,
                    create_if_not_exist=False,
                )
                for p in paths_list
            ]

            # get concentration values
            ds_list = [
                store.get_dataset(
                    dataset_name=species,
                    start_date=start+timedelta(days=fd),
                    end_date=end+timedelta(days=fd),
                    series_type=series_type,
                    step=step,
                )
                for store in store_list
            ]

            if len(ds_list) > 1:
                sim_ds = evt.dataset.Dataset(
                    stations=reduce(
                        np.union1d,
                        [d.data.columns for d in ds_list],
                    ),
                    start_date=start+timedelta(days=fd),
                    end_date=end+timedelta(days=fd),
                    species=species,
                    series_type=series_type,
                    step=step,
                )
                for ds in ds_list:
                    sim_ds.update_from_dataset(ds)
            else:
                sim_ds = ds_list[0]
            sim_ds_list.append(sim_ds)

        simulations = cls.from_dataset(
            model=model,
            ds_list=sim_ds_list,
            stations_idx=stations,
            step=step,
            correc_unit=correc_unit,
        )
        return simulations

    def drop_unrepresentative_stations(self, availability_ratio=0.75):
        """
        Drop stations with a certain rate of missing values.

        Modify Dataset object in place. If for every forecast days,
        the condition is not fulfilled, a station is dropped.

        Parameters
        ----------
        availability_ratio : float or None
            Minimal rate of data available on the period required to
            keep a station. If None, stations with only nan values
            are dropped.

        """
        dropped_sta = np.array([])
        for ds in self.datasets:
            ustations = ds.drop_unrepresentative_stations(availability_ratio,
                                                          drop=False)
            dropped_sta = np.union1d(dropped_sta, ustations)

        print(
            "{} stations dropped over {}.".format(
                dropped_sta.shape[0], self.datasets[0].data.shape[1]
            )
        )

        for ds in self.datasets:
            ds.data.drop(columns=dropped_sta, inplace=True)

    @deprecate_kwarg('forecastDay', 'forecast_day')
    def to_obs(self, forecast_day=0):
        """
        Transform Simulations object to Observations object.

        Parameters
        ----------
        forecast_day : int
            Forecast day for which to keep simulation data.

        """
        stations_df = pd.DataFrame(index=self.stations)
        obs = Observations(self.species, self.start_date,
                           self.end_date, stations_df, self.series_type,
                           self.forecast_horizon, step=self.step)
        obs.dataset.update_from_dataset(self.datasets[forecast_day])
        return obs

    @deprecate_kwarg('filePath', 'file_path')
    def to_csv(self, file_path):
        """
        Save timeseries dataframe as csv.

        Parameters
        ----------
        file_path : str
            File path with {forecast_day} instead of forecast day number.

        """
        for fd in range(self.forecast_horizon):
            self.sim_df[fd].to_csv(
                file_path.format(forecast_day=fd, forecastDay=fd),
                sep=',',
                na_rep='nan',
                float_format='%g',
                header=True,
                index=True,
                date_format=self.datasets[0].date_format,
            )

    @deprecate_kwarg('startDate', 'start_date')
    @deprecate_kwarg('endDate', 'end_date', stacklevel=3)
    def sub_period(self, start_date, end_date):
        """
        Build a new Simulations object define on a shorter period.

        Parameters
        ----------
        start_date : datetime.date
            Starting date of the new object.
        end_date : datetime.date
            Ending date of the new object.

        Returns
        -------
            Simulations object.

        """
        # check args integrity
        if start_date < self.start_date:
            raise evt.EvaltoolsError(
                f"start_date must be >= {self.start_date} !!!"
            )
        if end_date > self.end_date:
            raise evt.EvaltoolsError(
                f"end_date must be >= {self.end_date} !!!"
            )

        # select sub-period
        new_sim = self.__class__(
            start_date=start_date,
            end_date=end_date,
            stations=self.stations,
            species=self.species,
            model=self.model,
            series_type=self.series_type,
            forecast_horizon=self.forecast_horizon,
            step=self.step,
        )
        for fd in range(self.forecast_horizon):
            idx = np.logical_and(
                (self.datasets[fd].data.index.date >=
                 start_date+timedelta(days=fd)),
                (self.datasets[fd].data.index.date <=
                 end_date+timedelta(days=fd)))
            new_sim.datasets[fd].data = self.datasets[fd].data.loc[idx]
        return new_sim

    @deprecate_kwarg('filePath', 'file_path')
    def check_values(self, threshold, drop=False, file_path=None):
        """
        Check if observation values exceed a threshold.

        If there are values above the threshold, a message is printed
        and these values are set to nan if drop == True.

        Parameters
        ----------
        threshold : scalar
            Threshold value.
        drop : bool
            If True, values above the threshold are set to nan.
        file_path : None or str
            File path where to save the names of stations that exceed
            the threshold, the path must contain {forecast_day} instead
            of the forecast day number.

        """
        for fd in range(self.forecast_horizon):
            print(
                f"{self.model} threshold exceedances check "
                f"in forecast day = {fd}:"
            )
            self.datasets[fd].check_threshold(
                threshold,
                drop,
                file_path.format(forecastDay=fd, forecast_day=fd),
            )

    def daily_mean(self, availability_ratio=0.75):
        """
        Build Simulations object working on daily mean.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily mean.

        Returns
        -------
        evaluator.Simulations
            Simulations object with series_type = 'daily' and with data
            corresponding to the computed daily mean.

        """
        if self.series_type not in ['hourly']:
            print("Object's attribute 'series_type' must be 'hourly' " +
                  "to use this method !!!")
            return None

        new_sim = self.__class__(
            start_date=self.start_date, end_date=self.end_date,
            stations=self.stations, species=self.species,
            model=self.model, series_type='daily',
            forecast_horizon=self.forecast_horizon)
        for fd in range(self.forecast_horizon):
            new_sim.datasets[fd].update_from_dataset(
                evt.timeseries.daily_mean(
                    self.datasets[fd].data, availability_ratio))
        return new_sim

    def daily_max(self, availability_ratio=0.75):
        """
        Return Simulations object working on daily maximum.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily maximum.

        Returns
        -------
        evaluator.Simulations
            Simulations object with series_type = 'daily' and with data
            corresponding to the computed daily maximum.

        """
        if self.series_type not in ['hourly']:
            print("Object's attribute 'series_type' must be 'hourly' " +
                  "to use this method !!!")
            return None

        new_sim = self.__class__(
            start_date=self.start_date, end_date=self.end_date,
            stations=self.stations, species=self.species,
            model=self.model, series_type='daily',
            forecast_horizon=self.forecast_horizon)
        for fd in range(self.forecast_horizon):
            new_sim.datasets[fd].update_from_dataset(
                evt.timeseries.daily_max(
                    self.datasets[fd].data, availability_ratio))
        return new_sim

    def moving_average_daily_max(self, availability_ratio=0.75):
        """
        Compute the daily maximum of the 8-hour moving average.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available to compute the average for a
            given 8-hour window, and also minimal rate of values available
            for a given day to compute the daily maximum of the moving
            average. For example with availability_ratio=0.75, a daily maximum
            eight hours average can only be calculated if 18 eight hours
            average are available each of which requires 6 hourly values to
            be available.

        Returns
        -------
        evaluator.Simulations
            Simulations object with series_type = 'daily' and with data
            corresponding to the computed daily maximum of the moving average.

        """
        if self.series_type not in ['hourly'] and self.step == 1:
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "with one hour time step to use this method !!!"
            )
            return None

        new_sim = self.__class__(
            start_date=self.start_date, end_date=self.end_date,
            stations=self.stations, species=self.species,
            model=self.model, series_type='daily',
            forecast_horizon=self.forecast_horizon,
        )

        for fd in range(self.forecast_horizon):
            temp_df = copy.deepcopy(self.datasets[fd].data)
            # add nan for the day before since 1st window is 17 to 1 o'clock
            for new_row in temp_df.index[0:24]-timedelta(days=1):
                temp_df.loc[new_row] = np.nan
            temp_df.sort_index(inplace=True)
            temp_df = evt.timeseries.moving_average_daily_max(
                temp_df, availability_ratio)

            new_sim.datasets[fd].update_from_dataset(temp_df)

        return new_sim

    def filtered_series(self, availability_ratio=0.75):
        """
        Return Simulations object working on the filtered series.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available for a given hour to compute
            the daily filtered series for this hour in all days.

        Returns
        -------
        evaluator.Simulations
            Simulations object with series_type = 'hourly' and with data
            corresponding to the computed filtered series.

        """
        if self.series_type != 'hourly':
            print(
                "Object's attribute 'series_type' must be 'hourly' "
                "to use this method !!!"
            )
            return None

        new_sim = self.__class__(
            start_date=self.start_date, end_date=self.end_date,
            stations=self.stations, species=self.species,
            model=self.model, series_type='hourly',
            forecast_horizon=self.forecast_horizon)
        for fd in range(self.forecast_horizon):
            new_sim.datasets[fd].update_from_dataset(
                evt.timeseries.filtered_series(
                    self.datasets[fd].data, availability_ratio))
        return new_sim

    def normalized_series(self):
        """
        Return Simulations object with normalized series.

        This method normalizes each series of simulations by substracting
        its median and dividing by its interquartile range.

        Returns
        -------
        evaluator.Simulations
            Simulations object with series_type = 'hourly' and with data
            corresponding to the computed normalized series.

        """
        new_sim = self.__class__(
            start_date=self.start_date, end_date=self.end_date,
            stations=self.stations, species=self.species,
            model=self.model, series_type=self.series_type,
            forecast_horizon=self.forecast_horizon)
        for fd in range(self.forecast_horizon):
            new_sim.datasets[fd].update_from_dataset(
                evt.timeseries.normalized_series(self.datasets[fd].data))
        return new_sim


@deprecate_kwarg('inputFilePath', 'input_file_path')
def load(input_file_path):
    """
    Load an evaluator.Evaluator object.

    Load an evaluator.Evaluator object saved in binary format
    with evaluator.Evaluator.dump method.

    Parameters
    ----------
    input_file_path : str
        Path of the binary file to load.

    Returns
    -------
        evaluator.Evaluator object

    """
    with open(input_file_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, evt.evaluator.Evaluator):
        return obj
    else:
        raise evt.EvaltoolsError(
            f"*** {input_file_path} not an evaluator.Evaluator object !!!"
        )


@deprecate_attrs(
    start_date='startDate',
    end_date='endDate',
    forecast_horizon='forecastHorizon',
    series_type='seriesType',
    obs_df='obsDF',
    sim_df='simDF',
)
class Evaluator(object):
    """
    Class gathering observations and simulations of the studied case.

    An object of class :class:`~evaltools.evaluator.Simulations` will be
    specific to one species, one series type, one period and one model. This
    class contain several methods to compare simulations and observations.

    """

    def __init__(self, observations, simulations, color='k'):
        """
        Build an Evaluator object.

        Parameters
        ----------
        observations : evaluator.Observations object
            Observations used to construct the Evaluator object.
        simulations : evaluator.Simulations object
            Simulations used to construct the Evaluator object.
        color : str
            Default color that will be used in plotting functions.

        """
        if all([ds.data.empty for ds in simulations.datasets]):
            raise evt.EvaltoolsError(
                "Trying to construct Evaluator object "
                "with empty simulations !!!"
            )
        if observations.dataset.data.empty:
            raise evt.EvaltoolsError(
                "Trying to construct Evaluator object "
                "with empty observations !!!"
            )
        if observations.species != simulations.species:
            print(
                f"Warning: observations species {observations.species} != "
                f"simulations species {simulations.species}"
            )
        assert observations.start_date == simulations.start_date, \
            "observations.start_date != simulations.start_date !!!"
        assert observations.end_date == simulations.end_date, \
            "observations.end_date != simulations.end_date !!!"
        assert observations.forecast_horizon >= simulations.forecast_horizon, \
            "observations.forecast_horizon < simulations.forecast_horizon !!!"
        assert observations.series_type == simulations.series_type, \
            "observations.series_type != simulations.series_type !!!"

        if observations.series_type == 'hourly':
            if observations.step % simulations.step == 0 or \
                    simulations.step % observations.step == 0:
                step = max(simulations.step, observations.step)
            else:
                raise evt.EvaltoolsError(
                    "observations.step != simulations.step !!!"
                )
        else:
            step = None

        # only keep stations shared by obs and sim
        sta_idx = np.intersect1d(
            observations.stations.index,
            simulations.stations)
        stations = observations.stations.loc[sta_idx]

        self.observations = Observations(
            species=observations.species,
            start_date=observations.start_date,
            end_date=observations.end_date,
            stations=stations,
            series_type=observations.series_type,
            forecast_horizon=simulations.forecast_horizon,
            step=step,
            path=observations.path)
        self.observations.dataset.update_from_dataset(observations.dataset)
        self.observations.path = observations.path

        self.simulations = Simulations.from_dataset(
            model=simulations.model,
            ds_list=simulations.datasets,
            stations_idx=sta_idx,
            step=step,
            path=simulations.path)
        self.simulations.path = simulations.path

        # color attribute used in plotting function
        self.color = color

        # dictionary where to save score results
        self.results = {}

    @property
    def species(self):
        """Get the species date of the object."""
        return self.observations.species

    @property
    def start_date(self):
        """Get the starting date of the object."""
        return self.observations.start_date

    @property
    def end_date(self):
        """Get the ending of the object."""
        return self.observations.end_date

    @property
    def forecast_horizon(self):
        """Get the forecast horizon of the object."""
        return self.simulations.forecast_horizon

    @property
    def series_type(self):
        """Get the series type of the object."""
        return self.observations.series_type

    @property
    def model(self):
        """Get the model name of the object."""
        return self.simulations.model

    @property
    def stations(self):
        """Get the station list of the object."""
        return self.observations.stations

    @property
    def obs_df(self):
        """Get the observation data of the object."""
        return self.observations.dataset.data

    @property
    def sim_df(self):
        """Get the simulation data of the object."""
        return [self.simulations.datasets[fd].data
                for fd in range(self.forecast_horizon)]

    @property
    def step(self):
        """Get the time step of the object."""
        return self.observations.dataset.step

    @property
    def freq(self):
        """Get the time step of the object as a datetime.timedelta."""
        return self.observations.dataset.freq

    def colocate_nan(self):
        """Colocate missing values between observations and simulations."""
        for fd in range(self.forecast_horizon):
            start = self.start_date + timedelta(days=fd)
            end = self.end_date + timedelta(days=fd)
            idx_fd = (
                (self.obs_df.index.date >= start) &
                (self.obs_df.index.date <= end)
            )

            sim = self.get_sim(fd)
            idx_nan = np.logical_or(self.obs_df.loc[idx_fd].isna(), sim.isna())
            sim[idx_nan] = np.nan
            idx_nan = idx_nan.reindex(
                index=self.obs_df.index,
                columns=self.obs_df.columns,
                fill_value=False,
            )
            self.obs_df[idx_nan] = np.nan

    @deprecate_kwarg('outputFilePath', 'output_file_path')
    def dump(self, output_file_path):
        """
        Dump the evaluator.Evaluator object.

        Save an evaluator.Evaluator object in binary format.
        Use evaltools.evaluator.load function to get the object back.

        Parameters
        ----------
        output_file_path : str
            Path of the output binary file.

        """
        with open(output_file_path, 'wb') as f:
            pickle.dump(self, f)

    def summary(self):
        """Print a summary of the object."""
        print(f"Model: {self.model}")
        print(f"Species: {self.species}")
        if self.series_type == 'hourly':
            print(
                "Time step: {} hour{}".format(
                    self.step, 's' if self.step > 1 else '')
            )
        else:
            print(f"Time step: {self.series_type}")
        print("Period: {} - {}".format(
            self.start_date.strftime('%Y%m%d'),
            self.end_date.strftime('%Y%m%d')))
        print(f"Forecast horizon: {self.forecast_horizon}")
        print(f"Color: {self.color}")
        if self.simulations.path or self.observations.path:
            print("Paths :")
            print(f"- Sim : {self.simulations.path}")
            print(f"- Obs : {self.observations.path}")

    @deprecate_kwarg('startDate', 'start_date')
    @deprecate_kwarg('endDate', 'end_date', stacklevel=3)
    def sub_period(self, start_date, end_date):
        """
        Build a new Evaluator object define on a shorter period.

        Parameters
        ----------
        start_date : datetime.date
            Starting date of the new object.
        end_date : datetime.date
            Ending date of the new object.

        Returns
        -------
            Evaluator object.

        """
        return self.__class__(
            self.observations.sub_period(start_date, end_date),
            self.simulations.sub_period(start_date, end_date),
            color=self.color)

    def station_sub_list(self, station_list, inplace=True):
        """
        Drop stations not contained in the given list.

        Parameters
        ----------
        station_list : 1D array of str.
            List of stations to keep.
        inplace : bool
            If True, the Evaluator object is modified inplace,
            else, a new object is returned.

        """
        if inplace is True:
            obj = self
        else:
            obj = copy.deepcopy(self)
        obj.results = {}

        # check if station_list is included in self.stations.index
        stations = np.intersect1d(self.stations.index, station_list)
        missing_stations = np.array(station_list)[~np.in1d(station_list,
                                                           stations)]
        if missing_stations.shape[0] > 0:
            print(
                f"Warning: stations {missing_stations} from station_list "
                f"were not found in the original object !!!"
            )

        # drop unsought stations
        obj.observations.stations = obj.observations.stations.loc[stations]
        obj.observations.dataset.data = \
            obj.observations.dataset.data[stations]
        for fd in range(obj.forecast_horizon):
            obj.simulations.datasets[fd].data = \
                obj.simulations.datasets[fd].data[stations]

        if inplace is not True:
            return obj

    @deprecate_kwarg('minLon', 'min_lon')
    @deprecate_kwarg('maxLon', 'max_lon', stacklevel=3)
    @deprecate_kwarg('minLat', 'min_lat', stacklevel=4)
    @deprecate_kwarg('maxLat', 'max_lat', stacklevel=5)
    def sub_area(self, min_lon, max_lon, min_lat, max_lat, inplace=True):
        """
        Drop stations not contained within the given lat/lon boundaries.

        Parameters
        ----------
        min_lon, max_lon, min_lat, max_lat : scalars
            Lat/lon boundaries.
        inplace : bool
            If True, the Evaluator object is modified inplace,
            else, a new object is returned.

        """
        kept_stations_idx = reduce(
            np.logical_and,
            [self.stations['lat'] >= min_lat, self.stations['lat'] <= max_lat,
             self.stations['lon'] >= min_lon, self.stations['lon'] <= max_lon])
        if inplace is True:
            self.station_sub_list(
                self.stations.index[kept_stations_idx],
                inplace=inplace)
        else:
            return self.station_sub_list(
                self.stations.index[kept_stations_idx],
                inplace=inplace)

    def select_countries(self, countries, inplace=True):
        """
        Only keep stations within certain countries.

        This method assumes that the first caracters of a station code
        refer to the station country.

        Parameters
        ----------
        countries : 1D array of str.
            List of first letters of station codes to keep
            (e.g. ['FRA', 'IT', 'AUT', 'ES']).
        inplace : bool
            If True, the Evaluator object is modified inplace,
            else, a new object is returned.

        """
        kept_stations = [sta for sta in self.stations.index
                         if any([re.match(cn, sta) for cn in countries])]
        if inplace is True:
            self.station_sub_list(kept_stations, inplace=inplace)
        else:
            return self.station_sub_list(kept_stations, inplace=inplace)

    @deprecate_kwarg('obsOutputFile', 'obs_output_file')
    @deprecate_kwarg('simOutputFile', 'sim_output_file', stacklevel=3)
    def time_series_to_csv(self, obs_output_file, sim_output_file):
        """
        Save timeseries dataframes as csv.

        Parameters
        ----------
        obs_output_file : str
            File path where to save observed timeseries.
        sim_output_file : str
            File path where to save simulated timeseries. The path name
            must contain {forecast_day} instead of the forecast day number.

        """
        self.obs_df.to_csv(
            obs_output_file, sep=',', na_rep='nan',
            float_format='%g', header=True, index=True,
            date_format='%Y%m%d%H')
        for forecast_day in range(self.forecast_horizon):
            self.sim_df[forecast_day].to_csv(
                sim_output_file.format(
                    forecastDay=forecast_day,
                    forecast_day=forecast_day,
                ),
                sep=',', na_rep='nan', float_format='%g', header=True,
                index=True, date_format='%Y%m%d%H')

    @deprecate_kwarg('forecastDay', 'forecast_day')
    def get_obs(self, forecast_day, start_end=None):
        """
        Get observations according to the forecast day.

        Parameters
        ----------
        forecast_day : int
            Forecast day for which to get observations.
        start_end : None or list of two datetime.date objects
            Dates between which getting the data.

        Returns
        -------
        pandas.DataFrame
            Observations for the time period corresponding to
            the forecast day.

        """
        if start_end is not None:
            start = start_end[0] + timedelta(days=forecast_day)
            end = start_end[1] + timedelta(days=forecast_day)
        else:
            start = self.start_date + timedelta(days=forecast_day)
            end = self.end_date + timedelta(days=forecast_day)

        idx = ((self.obs_df.index.date >= start) &
               (self.obs_df.index.date <= end))

        return self.obs_df.loc[idx]

    @deprecate_kwarg('forecastDay', 'forecast_day')
    def get_sim(self, forecast_day, start_end=None):
        """
        Get simulations according to the forecast day.

        Parameters
        ----------
        forecast_day : int
            Forecast day for which to get simulations.
        start_end : None or list of two datetime.date objects
            Dates between which getting the data.

        Returns
        -------
        pandas.DataFrame
            Simulations corresponding to the forecast day.

        """
        if start_end is not None:
            start = start_end[0] + timedelta(days=forecast_day)
            end = start_end[1] + timedelta(days=forecast_day)

            idx = ((self.sim_df[forecast_day].index.date >= start) &
                   (self.sim_df[forecast_day].index.date <= end))

            return self.sim_df[forecast_day].loc[idx]

        return self.sim_df[forecast_day]

    def remove_negvalues(self, rpl_value='nan'):
        """
        Replace negative values.

        Replace negative values of observations and simulations data by
        another value.

        Parameters
        ----------
        rpl_value : scalar or 'nan'
            Replacement value for negative values.

        """
        rpl = np.nan if rpl_value == 'nan' else rpl_value
        self.observations.dataset.data.where(
            self.observations.dataset.data >= 0,
            rpl,
            inplace=True)
        for fd in range(self.forecast_horizon):
            self.simulations.datasets[fd].data.where(
                self.simulations.datasets[fd].data >= 0,
                rpl,
                inplace=True)

    def replace_value(self, needle='nan', replace_value=-999):
        """
        Replace a choosen value.

        Replace a choosen value in observations and simulations data by
        another value.

        Parameters
        ----------
        needle : scalar or 'nan'
            Value to be replaced.
        replace_value : scalar or 'nan'
            Replacement value for needle.

        """
        rpl = np.nan if replace_value == 'nan' else replace_value
        if needle != 'nan':
            self.observations.dataset.data.where(
                self.observations.dataset.data == needle,
                rpl,
                inplace=True)
            for fd in range(self.forecast_horizon):
                self.simulations.datasets[fd].data.where(
                    self.simulations.datasets[fd].data == needle,
                    rpl,
                    inplace=True)
        else:
            self.observations.dataset.data.where(
                ~np.isnan(self.observations.dataset.data),
                rpl, inplace=True)
            for fd in range(self.forecast_horizon):
                self.simulations.datasets[fd].data.where(
                    ~np.isnan(self.simulations.datasets[fd].data),
                    rpl,
                    inplace=True)

    @deprecate_kwarg('outputFile', 'output_file')
    def temporal_scores(
            self, score_list, output_file=None, availability_ratio=0.75):
        """
        Compute temporal scores per station for each forecast day.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        output_file : str
            File where to save the result. The file name must contain
            {forecastDay} instead of the forecast day number. If None,
            result is not saved in csv.
        availability_ratio : float
            Minimum required rate of available data over the period to
            calculate the scores.

        Returns
        -------
        dictionary
            Dictionary with one key per forecast day ('D0', 'D1', ...)
            containing pandas.DataFrames with station names as index and one
            column per computed score.

        """
        res = {}
        for forecast_day in range(self.forecast_horizon):
            obs = self.get_obs(forecast_day)
            sim = self.get_sim(forecast_day)
            stats_df_sta = evt.scores.stats2d(
                obs, sim, score_list,
                axis=0,
                threshold=float(availability_ratio),
                keep_nan=True,
            )
            if output_file is not None:
                stats_df_sta.to_csv(
                    output_file.format(
                        forecastDay=forecast_day,
                        forecast_day=forecast_day,
                    ),
                    sep=' ', na_rep='nan', float_format='%g',
                    header=True, index=True,
                    date_format='%Y%m%d%H',
                )
            res["D{}".format(str(forecast_day))] = stats_df_sta
        return res

    def average_temporal_scores(
            self, score_list, availability_ratio=0.75,
            averaging='median', min_nb_sta=10):
        """
        Compute average temporal scores for each forecast day.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        availability_ratio : float
            Minimum required rate of available data over the period to
            calculate the scores.
        averaging : str
            Type of score averaging selected from 'mean' or 'median'.
        min_nb_sta : int
            Minimum required number of stations to compute the average of
            the scores.

        Returns
        -------
        pandas.DataFrame
            Dataframe with one column per score and one line per forecast
            day ('D0', 'D1', ...).

        """
        if averaging == 'mean':
            average = pd.DataFrame.mean
        elif averaging == 'median':
            average = pd.DataFrame.median
        else:
            raise evt.EvaltoolsError(
                "averaging argument must be either equal to 'mean' "
                "or 'median'."
            )

        scores = self.temporal_scores(
            score_list=score_list,
            availability_ratio=availability_ratio,
        )

        res = pd.DataFrame()
        for fd in range(self.forecast_horizon):
            scores_fd = scores[f'D{fd}']
            idx_enough_values = (
                np.sum(~scores_fd.isna(), axis=0) >= min_nb_sta
            )
            scores_fd = average(scores_fd)
            scores_fd[~idx_enough_values] = np.nan
            res = pd.concat(
                [res, scores_fd.to_frame(name=f'D{fd}').T],
                sort=False,
            )

        return res

    @deprecate_kwarg('outputFile', 'output_file')
    @deprecate_kwarg('forecastDay', 'forecast_day', stacklevel=3)
    def conc_scores(self, score_list, conc_range, output_file=None,
                    min_nb_val=10, based_on='obs', forecast_day=0):
        """
        Compute scores for an interval of concentration values.

        For each station, scores are computed keeping only times where the
        observed values (if based_on='obs') or the simulated values (if
        based_on='sim') fall within conc_range.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        conc_range : list of two scalars
            Interval of concentrations to keep to compute the scores.
        output_file : str
            File where to save the result. The file name can contain
            {forecast_day} instead of the forecast day number. If None,
            result is not saved in csv.
        min_nb_val : int
            Minimal number of (obs, sim) couple required for a score to be
            computed.
        based_on : str
            If 'sim', the concentration interval is determined from simulation
            data. Else ('obs') it is determined with observations.
        forecast_day : int
            Integer corresponding to the chosen forecast day.

        Returns
        -------
        pandas.DataFrame
            Dataframe with one column per computed score.
        scalar
            Number of values kept to compute the scores.

        """
        fd = forecast_day

        obs = self.get_obs(fd)
        sim = self.get_sim(fd)

        if based_on == 'obs':
            obs = obs[(conc_range[0] <= obs) & (obs < conc_range[1])]
        elif based_on == 'sim':
            sim = sim[(conc_range[0] <= sim) & (sim < conc_range[1])]
        else:
            raise evt.EvaltoolsError(
                "based_on argument must be 'obs' or 'sim'")

        if based_on == 'obs':
            obs = obs.loc[:, (~obs.isnull().all()) & (~sim.isnull().all())]
            sim = sim[obs.columns]
            sim = sim[~obs.isna()]
        else:
            sim = sim.loc[:, (~obs.isnull().all()) & (~sim.isnull().all())]
            obs = obs[sim.columns]
            obs = obs[~sim.isna()]

        stats_df_sta = evt.scores.stats2d(
            obs, sim, score_list,
            axis=0,
            threshold=int(min_nb_val),
            keep_nan=True,
        )

        if output_file is not None:
            stats_df_sta.to_csv(
                output_file.format(forecastDay=fd, forecast_day=fd),
                sep=' ', na_rep='nan', float_format='%g',
                header=True, index=True,
                date_format='%Y%m%d%H',
            )

        return stats_df_sta, obs.count().sum()  # , stats_df_sta.count()[0]

    @deprecate_kwarg('filePath', 'file_path')
    @deprecate_kwarg('forecastDay', 'forecast_day', stacklevel=3)
    def quarterly_score(
            self, file_path, score='RMSE', forecast_day=0,
            score_type='temporal', averaging='median',
            availability_ratio=0.75, min_nb_sta=10):
        """
        Calculate an average score.

        The period of the object must corresponds to a valid quarter.

        Parameters
        ----------
        file_path : str
            File where to find previous values and save the result.
        score : str
            The score to process.
        forecast_day : int
            Integer corresponding to the chosen forecast day used for
            computation.
        score_type : str
            Computing method selected from 'temporal', 'spatial' or
            'spatiotemporal'.
        averaging : str
            Type of score averaging choosen among 'mean' or 'median'.
            This parameter is ignored if score_type == spatiotemporal.
        availability_ratio : float
            Minimum required rate of available data over the period to
            calculate the score if score_type is 'temporal, or to calculate
            the average of the score if score_type is 'spatial'.
            This parameter is ignored if score_type is 'spatiotemporal'.
        min_nb_sta : int
            Minimal number of values required to compute the average of
            the score if score_type is 'temporal, or to compute the score
            itself if score_type is 'spatial' or 'spatiotemporal'.

        Returns
        -------
            The median for the computed score.

        """
        if averaging == 'mean':
            average = np.nanmean
        elif averaging == 'median':
            average = np.nanmedian
        else:
            raise evt.EvaltoolsError(
                "averaging argument must be either equal to 'mean' "
                "or 'median'."
            )

        if score_type == 'temporal':
            stats = self.temporal_scores(
                score_list=[score],
                availability_ratio=availability_ratio,
            )['D{}'.format(forecast_day)][score]
        elif score_type == 'spatial':
            stats = self.spatial_scores(
                score_list=[score],
                min_nb_sta=min_nb_sta,
            )['D{}'.format(forecast_day)][score]
        elif score_type == 'spatiotemporal':
            res = self.spatiotemporal_scores(
                score_list=[score],
                forecast_days=[forecast_day],
                threshold=int(min_nb_sta),
            ).values[0, 0]
        else:
            raise evt.EvaltoolsError(
                "score_type argument must be either equal to "
                "'temporal', 'spatial' or 'spatiotemporal'."
            )

        if isfile(file_path):
            df = pd.read_csv(file_path, sep=' ', index_col=0)
        else:
            df = pd.DataFrame(columns=[score])

        quarter = evt.quarter.Quarter(self.start_date, self.end_date)
        idx = quarter.__repr__()

        if score_type == 'temporal':
            nb_val = np.sum(~stats.isna())
            nb_val = np.sum(~stats.isna())
            if nb_val < min_nb_sta:
                res = np.nan
            else:
                res = average(stats)
        elif score_type == 'spatial':
            nb_val = np.sum(~stats.isna())
            if nb_val/stats.shape[0] < availability_ratio:
                res = np.nan
            else:
                res = average(stats)

        df.loc[idx] = res
        df.to_csv(
            file_path, sep=' ', na_rep='nan', float_format='%g',
            header=True, index=True, date_format='%Y%m%d%H',
        )
        return res

    @deprecate_kwarg('filePath', 'file_path')
    @deprecate_kwarg('forecastDay', 'forecast_day', stacklevel=3)
    def quarterly_median_score(
            self, file_path, score='RMSE', forecast_day=0,
            availability_ratio=0.75, min_nb_sta=1):
        """
        Compute median on station scores.

        Parameters
        ----------
        file_path : str
            File where to find previous values and save the result.
        score : str
            The score to process.
        forecast_day : int
            Integer corresponding to the chosen forecast_day used for
            computation.
        availability_ratio : float
            Minimal rate of data available on the period required per
            forecast day to compute the scores.
        min_nb_sta : int
            Minimal number of station values available required to compute
            the median.

        Returns
        -------
            The median for the computed score.

        """
        stats = self.temporal_scores(
            score_list=[score],
            availability_ratio=availability_ratio,
        )['D{}'.format(forecast_day)][score]

        if isfile(file_path):
            df = pd.read_csv(file_path, sep=' ', index_col=0)
        else:
            df = pd.DataFrame(columns=[score])

        quarter = evt.quarter.Quarter(self.start_date, self.end_date)
        idx = quarter.__repr__()

        nb_sta = np.sum(~stats.isna())
        if nb_sta < min_nb_sta:
            res = np.nan
        else:
            res = np.nanmedian(stats)

        df.loc[idx] = res
        df.to_csv(file_path, sep=' ', na_rep='nan', float_format='%g',
                  header=True, index=True, date_format='%Y%m%d%H')
        return res

    @deprecate_kwarg('outputFile', 'output_file')
    def spatial_scores(self, score_list, output_file=None, min_nb_sta=10):
        """
        Compute spatial scores per time step.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        output_file : str.
            File where to save the result. The file name must contain
            {forecast_day} instead of the forecast day number. If None,
            result is not saved in csv.
        min_nb_sta : int
            Minimal number of station values available in both obs and
            sim required per datetime to compute the scores.

        Returns
        -------
        dictionary
            Dictionary with one key per forecast day ('D0', 'D1', ...)
            containing pandas.DataFrames with datetime index and one column
            per computed score.

        """
        try:
            # test if these score are already computed
            assert (
                self.results['spatial_scores']['args'] ==
                {'score_list': score_list, 'min_nb_sta': min_nb_sta}
            )
            res = self.results['spatial_scores']['result']

        except (AssertionError, KeyError):
            res = {}
            for fd in range(self.forecast_horizon):
                obs = self.get_obs(fd)
                sim = self.get_sim(fd)
                stats_df_sta = evt.scores.stats2d(
                    obs, sim, score_list,
                    axis=1,
                    threshold=int(min_nb_sta),
                )
                if output_file is not None:
                    stats_df_sta.to_csv(
                        output_file.format(forecastDay=fd, forecast_day=fd),
                        sep=' ', na_rep='nan', header=True,
                        float_format='%g', index=True,
                        date_format='%Y%m%d%H',
                    )
                res["D{}".format(str(fd))] = stats_df_sta
            self.results['spatial_scores'] = {
                'args': {'score_list': score_list, 'min_nb_sta': min_nb_sta},
                'result': res,
            }

        return res

    def average_spatial_scores(
            self, score_list, min_nb_sta=10,
            averaging='median', availability_ratio=0.75):
        """
        Compute average spatial scores for each forecast day.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        min_nb_sta : int
            Minimum required number of stations to compute the scores.
        averaging : str
            Type of score averaging selected from 'mean' or 'median'.
        availability_ratio : float
            Minimum required rate of available data over the period to
            calculate the average of the scores.

        Returns
        -------
        pandas.DataFrame
            Dataframe with one column per score and one line per forecast
            day ('D0', 'D1', ...).

        """
        if averaging == 'mean':
            average = pd.DataFrame.mean
        elif averaging == 'median':
            average = pd.DataFrame.median
        else:
            raise evt.EvaltoolsError(
                "averaging argument must be either equal to 'mean' "
                "or 'median'."
            )

        scores = self.spatial_scores(
            score_list=score_list,
            min_nb_sta=min_nb_sta,
        )

        res = pd.DataFrame()
        for fd in range(self.forecast_horizon):
            scores_fd = scores[f'D{fd}']
            idx_enough_values = (
                np.sum(~scores_fd.isna(), axis=0) >=
                availability_ratio*scores_fd.shape[0]
            )
            scores_fd = average(scores_fd)
            scores_fd[~idx_enough_values] = np.nan
            res = pd.concat(
                [res, scores_fd.to_frame(name=f'D{fd}').T],
                sort=False,
            )

        return res

    @deprecate_kwarg('outputFile', 'output_file')
    def mean_time_scores(
            self, score_list, output_file=None, min_nb_sta=10,
            availability_ratio=0.75):
        """
        Compute the mean of time scores for each forecast time.

        Parameters
        ----------
        score_list : list of str
            List of computed scores.
        output_file : str
            File where to save the result. The file name must contain
            {forecast_day} instead of the forecast day number. If None,
            result is not saved in csv.
        min_nb_sta : int
            Minimal number of station values available in both obs and sim
            required per datetime to compute the scores (before applying mean
            for each forecast time).
        availability_ratio : float
            Minimal rate of data (computed scores per time) available on the
            period required per forecast time to compute the mean scores.

        Returns
        -------
        dictionary
            Dictionary, with one key per score, that contains lists of the
            means for each forecast time. For example if the forecast horizon
            is 4, the lists will be of length 96.

        """
        if self.series_type not in ['hourly']:
            raise evt.EvaltoolsError("Object's attribute 'series_type' must " +
                                     "be 'hourly' to use this method !!!")

        # find number of values per day
        unique, counts = np.unique(self.obs_df.index.date, return_counts=True)
        nval = counts[0]

        # scores computing
        stats = self.spatial_scores(score_list, min_nb_sta=min_nb_sta)

        res = {}
        nb_days = (self.end_date - self.start_date).days + 1
        for sc in score_list:
            y = []
            for fd in range(self.forecast_horizon):
                matrix = stats["D{}".format(str(fd))][sc].values.reshape(
                    nb_days, nval)
                idx_valid = (
                    np.sum(~np.isnan(matrix), axis=0) >=
                    availability_ratio*matrix.shape[0]
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Mean of empty slice')
                    means = np.nanmean(matrix, axis=0)
                means[~idx_valid] = np.nan
                y.extend(list(means))
            res[sc] = y
        if output_file is not None:
            pd.DataFrame(res).to_csv(output_file, sep=' ', na_rep='nan',
                                     float_format='%g', header=True,
                                     index=True, date_format='%Y%m%d%H')

        return res

    @deprecate_kwarg('outputFile', 'output_file')
    def temporal_ft_scores(
            self, score_list, availability_ratio=0.75, output_file=None,
            coords=False):
        """
        Compute forecast time scores for each station.

        Only available for hourly time step data.

        Parameters
        ----------
        score_list : list of str
            List of scores to compute.
        availability_ratio : float
            Minimal rate of data available on the period required per
            forecast time to compute the scores for each station.
        output_file : str
            File where to save the result. If None, result is not saved in
            csv. The file name must contain {score} instead of the score
            name. A file in created for each score.
        coords : bool
            If True, lat/lon coordinates are copied in the output.

        Returns
        -------
        dictionary
            Dictionary, with one key per score, that contains
            pandas.DataFrame with one row per station and one column per
            forecast time except the first two columns that are latitude
            and longitude.

        """
        try:
            assert (
                self.results['temp_ft_sc']['args'] ==
                {
                    'score_list': score_list,
                    'availability_ratio': availability_ratio,
                    'coords': coords,
                }
            )
            res = self.results['temp_ft_sc']['result']

        except (AssertionError, KeyError):
            if self.series_type not in ['hourly']:
                print(
                    "Object's attribute 'series_type' must be 'hourly' "
                    "to use this method !!!"
                )
                return None
            res = {}
            for sc in score_list:
                if coords:
                    try:
                        res[sc] = self.stations[['lat', 'lon']]
                    except KeyError:
                        res[sc] = pd.DataFrame(
                            index=self.stations.index,
                            columns=['lat', 'lon'],
                        )
                    res[sc].columns = pd.MultiIndex.from_product(
                        [['coord'], ['lat', 'lon']]
                    )
                else:
                    res[sc] = pd.DataFrame(index=self.stations.index)
            for day in range(self.forecast_horizon):
                obs = self.get_obs(day)
                sim = self.get_sim(day)
                daily_stats = evt.scores.ft_stats(
                    obs, sim, score_list, availability_ratio, day,
                    step=self.step,
                )
                for sc in score_list:
                    if coords:
                        midx = pd.MultiIndex.from_product(
                            [['D{}'.format(day)], range(0, 24, self.step)]
                        )
                        daily_stats[sc].columns = midx
                    res[sc] = pd.concat(
                        [res[sc], daily_stats[sc]],
                        axis=1, sort=False,
                    )
            if output_file is not None:
                for sc in score_list:
                    file_path = output_file.format(score=sc)
                    res[sc].insert(0, ('station', 'station'), res[sc].index)
                    res[sc].to_csv(
                        file_path, sep=',', na_rep='nan',
                        float_format='%g', header=True,
                        index=False, date_format='%Y%m%d%H',
                    )
            self.results['temp_ft_sc'] = {
                'args': {
                    'score_list': score_list,
                    'availability_ratio': availability_ratio,
                    'coords': coords,
                },
                'result': res,
            }

        return res

    @deprecate_kwarg('outputFile', 'output_file')
    def median_station_scores(
            self, score_list, availability_ratio=0.75,
            min_nb_sta=10, output_file=None):
        """
        Compute median station scores for each forecast time.

        Parameters
        ----------
        score_list : list of str
            List of scores to compute.
        availability_ratio : float
            Minimal rate of data available on the period required per
            forecast time to compute the scores for each station.
        min_nb_sta : int
            Minimal number of station values available in both obs and sim
            required to compute the median.
        output_file : str
            File where to save the result. If None, result is not saved in
            csv. The file name must contain {score} instead of the score
            name. A file is created for each score.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per forecast time and one column
            per score.

        """
        return self.average_ft_scores(
            score_list=score_list,
            availability_ratio=availability_ratio,
            min_nb_sta=min_nb_sta,
            output_file=output_file,
            score_type='temporal',
            averaging='median',
        )

    def average_ft_scores(
            self, score_list, availability_ratio=0.75, min_nb_sta=10,
            output_file=None, score_type='temporal', averaging='median'):
        """
        Compute average scores for each forecast time.

        Parameters
        ----------
        score_list : list of str
            List of scores to compute.
        availability_ratio : float
            Minimum required rate of available data over the period to
            calculate the score if score_type is 'temporal, or to calculate
            the average of the score if score_type is 'spatial'.
        min_nb_sta : int
            Minimal number of stations required to compute the average of
            the score if score_type is 'temporal, or to compute the score
            itself if score_type is 'spatial'.
        output_file : str
            File where to save the result. If None, result is not saved in
            csv. The file name must contain {score} instead of the score
            name. A file is created for each score.
        score_type : str
            Computing method selected from 'temporal' or 'spatial'.
        averaging : str
            Type of score averaging choosen among 'mean' or 'median'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per forecast time and one column
            per score.

        """
        if averaging == 'mean':
            average = pd.DataFrame.mean
        elif averaging == 'median':
            average = pd.DataFrame.median
        else:
            raise evt.EvaltoolsError(
                "averaging argument must be either equal to 'mean' "
                "or 'median'."
            )

        if score_type == 'temporal':
            if self.series_type == 'hourly':
                df_dict = self.temporal_ft_scores(
                    score_list,
                    availability_ratio=availability_ratio,
                )
                av_scores = {}
                for sc in score_list:
                    av_scores[sc] = average(df_dict[sc], axis=0)
                res = pd.DataFrame(av_scores)
                idx_not_enough_values = np.sum(
                    ~(list(df_dict.values())[0]).isna(),
                    axis=0,
                ).values < min_nb_sta
            elif self.series_type == 'daily':
                df_dict = self.temporal_scores(
                    score_list, availability_ratio=availability_ratio,
                )
                av_scores = {}
                for fd in range(self.forecast_horizon):
                    av_scores[fd] = average(df_dict[f'D{fd}'])
                res = pd.DataFrame(av_scores).T
                idx_not_enough_values = [
                    np.sum(~df_dict[f'D{fd}'][score_list[0]].isna()) <
                    min_nb_sta
                    for fd in range(self.forecast_horizon)
                ]

            res[idx_not_enough_values] = np.nan

        elif score_type == 'spatial':
            nb_days = (self.end_date - self.start_date).days + 1
            if self.series_type == 'hourly':
                # find number of values per day
                unique, counts = np.unique(
                    self.obs_df.index.date,
                    return_counts=True,
                )
                nval = counts[0]

                # scores computing
                stats = self.spatial_scores(score_list, min_nb_sta=min_nb_sta)

                res = {}
                for sc in score_list:
                    y = []
                    for fd in range(self.forecast_horizon):
                        matrix = stats["D{}".format(str(fd))][sc].values
                        matrix = matrix.reshape(nb_days, nval)
                        idx_valid = (
                            np.sum(~np.isnan(matrix), axis=0) >=
                            availability_ratio*matrix.shape[0]
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                'ignore', 'Mean of empty slice',
                            )
                            av_score = average(pd.DataFrame(matrix), axis=0)
                        av_score[~idx_valid] = np.nan
                        y.extend(list(av_score))
                    res[sc] = y

                x_max = 24*self.forecast_horizon
                idx = range(0, x_max, self.step)
                res = pd.DataFrame(res, index=idx)

            elif self.series_type == 'daily':
                df_dict = self.spatial_scores(
                    score_list,
                    min_nb_sta=min_nb_sta,
                )
                av_scores = {}
                for fd in range(self.forecast_horizon):
                    av_scores[fd] = average(df_dict[f'D{fd}'])
                res = pd.DataFrame(av_scores).T
                idx_not_enough_values = [
                    np.sum(~df_dict[f'D{fd}'][score_list[0]].isna()) <
                    availability_ratio*nb_days
                    for fd in range(self.forecast_horizon)
                ]
                res[idx_not_enough_values] = np.nan

        else:
            raise evt.EvaltoolsError(
                "score_type argument must be either equal to "
                "'temporal' or 'spatial'."
            )

        if output_file is not None:
            pd.DataFrame(res).to_csv(
                output_file, sep=' ', na_rep='nan',
                float_format='%g', header=True,
                index=True, date_format='%Y%m%d%H',
            )

        return res

    @deprecate_kwarg('outputFile', 'output_file')
    def spatiotemporal_scores(
            self, score_list, forecast_days='all', threshold=.75,
            output_file=None):
        """
        Compute spatio-temporal scores.

        Scores are computed using all data available for a given forecast day
        (values for all station at all times are considered as a simple
        1D array).

        Parameters
        ----------
        score_list : list of str
            List of scores to compute.
        forecast_days : 'all' or list of int
            Forcast days used for computation. The returned DataFrame will
            contain one row per forcast day.
        threshold : int or float
            Minimal number (if type(threshold) is int) or minimal rate
            (if type(threshold) is float) of data available in both obs and
            sim required to compute the scores.
        output_file : str
            File where to save the result. If None, result is not saved
            in csv.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one column per score and one row per
            forecast day.

        """
        res = pd.DataFrame()
        if forecast_days == 'all':
            forecast_days = range(self.forecast_horizon)

        for day in forecast_days:
            obs = self.get_obs(day).values.flatten()
            sim = self.get_sim(day).values.flatten()
            # define threshold in terms of number of obs instead of ratio
            if isinstance(threshold, float):
                if threshold >= 1.:
                    print("*** Warning: threshold of type float >= 1")
                thresh = threshold*obs.shape[0]
            else:
                thresh = threshold
            df = pd.DataFrame(
                evt.scores.stats2d_numpy(
                    obs.reshape(obs.size, 1),
                    sim.reshape(sim.size, 1),
                    score_list,
                    threshold=thresh,
                ),
                index=["D{}".format(day)],
                columns=score_list,
            )
            res = pd.concat((res, df))

        if output_file is not None:
            res.to_csv(
                output_file, sep=' ', na_rep='nan',
                float_format='%g', header=True, index=True,
            )
        return res

    def daily_mean(self, availability_ratio=0.75):
        """
        Build Evaluator object working on daily mean.

        This method computes the daily mean of observations and simulations
        of the current object.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily mean.

        Returns
        -------
        evaluator.Evaluator
            Evaluator object with series_type = 'daily' and with data
            corresponding to the computed daily mean.

        """
        return self.__class__(self.observations.daily_mean(availability_ratio),
                              self.simulations.daily_mean(availability_ratio),
                              color=self.color)

    def daily_max(self, availability_ratio=0.75):
        """
        Return Evaluator object working on daily maximum.

        This method compute the daily maximum of observations and simulations
        of the current object and returns a new Evaluator object with
        series_type = 'daily' and with data corresponding to the computed
        daily maxima.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of data available in a day required
            to compute the daily maximum.

        Returns
        -------
        evaluator.Evaluator
            Evaluator object with series_type = 'daily' and with data
            corresponding to the computed daily maximum.

        """
        return self.__class__(self.observations.daily_max(availability_ratio),
                              self.simulations.daily_max(availability_ratio),
                              color=self.color)

    def moving_average_daily_max(self, availability_ratio=0.75):
        """
        Compute the daily maximum of the moving average.

        This method compute the daily maximum of the moving average
        for observations and simulations of the current object and returns
        a new Evaluator object.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available to compute the average for a
            given 8-hour window, and also minimal rate of values available
            for a given day to compute the daily maximum of the moving
            average. For example with availability_ratio=0.75, a daily
            maximum eight hours average can only be calculated if 18 eight
            hours average are available each of which requires 6 hourly
            values to be available.

        Returns
        -------
        evaluator.Evaluator
            Evaluator object with series_type = 'daily' and with data
            corresponding to the computed daily maximum of the moving average.

        """
        return self.__class__(
            self.observations.moving_average_daily_max(availability_ratio),
            self.simulations.moving_average_daily_max(availability_ratio),
            color=self.color)

    def filtered_series(self, availability_ratio=0.75):
        """
        Return Evaluator object working on the filtered series.

        This method compute the filtered series of observations and
        simulations of the current object and returns a new Evaluator object
        with series_type = 'hourly' and with data corresponding to the computed
        filtered series.

        Parameters
        ----------
        availability_ratio : float
            Minimal rate of values available for a given hour to compute the
            daily filtered series for this hour in all days.

        Returns
        -------
        evaluator.Evaluator
            Evaluator object with series_type = 'hourly' and with data
            corresponding to the computed filtered series.

        """
        return self.__class__(
            self.observations.filtered_series(availability_ratio),
            self.simulations.filtered_series(availability_ratio),
            color=self.color)

    def normalized_series(self):
        """
        Return Evaluator object with normalized series.

        This method normalizes each series of observations and simulations
        by substracting its median and dividing by its interquartile range.

        Returns
        -------
        evaluator.Evaluator
            Evaluator object with series_type = 'hourly' and with data
            corresponding to the computed normalized series.

        """
        return self.__class__(self.observations.normalized_series(),
                              self.simulations.normalized_series(),
                              color=self.color)

    @deprecate_kwarg('outputFile', 'output_file')
    def contingency_table(self, threshold, output_file):
        """
        Contingency table.

        The table is computed from daily data for each forecast day.
        Before computed the table, values for every stations and every
        days of the period are concatenated. Tables corresponding to
        the different forecast days are stored in the same file.

        Parameters
        ----------
        threshold : scalar
            Threshold value.
        output_file : str
            File where to save the result.

        """
        if self.series_type not in ['daily']:
            print("*** Object's attribute 'series_type' must be 'daily' " +
                  "to use this method !!!")
            return

        with open(output_file, 'w') as f:
            for fd in range(self.forecast_horizon):
                obs = self.get_obs(fd)
                obs = obs.values.flatten()
                sim = self.get_sim(fd)
                sim = sim.values.flatten()
                res = evt.scores.contingency_table(obs, sim, thr=threshold)
                res = pd.DataFrame(
                    res,
                    index=["obs>s", "obs<s", "total"],
                    columns=["sim>s", "sim<s", "total"]).astype(int)
                f.write("D{}\n".format(fd) + res.T.to_string() + '\n\n')

    @deprecate_kwarg('outputFile', 'output_file')
    def obs_exceedances(self, threshold, output_file):
        """
        Look for exceedances in observed time series.

        Parameters
        ----------
        threshold : scalar
            Threshold value.
        output_file : str
            File where to save the result.

        """
        ts = self.obs_df.stack()
        idx = ts > threshold
        res = ts.loc[idx]
        with open(output_file, 'w') as f:
            f.write(res.to_string())

    @deprecate_kwarg('outputFile', 'output_file')
    def sim_exceedances(self, threshold, output_file):
        """
        Look for exceedances in simulated time series.

        Parameters
        ----------
        threshold : scalar
            Threshold value.
        output_file : str
            File where to save the result. The file name must contain
            {forecast_day} instead of the forecast day number.

        """
        for fd in range(self.forecast_horizon):
            ts = self.get_sim(fd).stack()
            idx = ts > threshold
            res = ts.loc[idx]
            file_path = output_file.format(forecastDay=fd, forecast_day=fd)
            with open(file_path, 'w') as f:
                f.write(res.to_string())


Observations.fromTimeSeries = deprecate(
    'fromTimeSeries',
    Observations.from_time_series,
)
Observations.fromDataset = deprecate(
    'fromDataset',
    Observations.from_dataset,
)
Observations.persistenceModel = deprecate(
    'persistenceModel',
    Observations.persistence_model,
)
Observations.subPeriod = deprecate(
    'subPeriod',
    Observations.sub_period,
)
Observations.simVSobs = deprecate(
    'simVSobs',
    Observations.sim_vs_obs,
)
Observations.dailyMean = deprecate(
    'dailyMean',
    Observations.daily_mean,
)
Observations.dailyMax = deprecate(
    'dailyMax',
    Observations.daily_max,
)
Observations.movingAverageDailyMax = deprecate(
    'movingAverageDailyMax',
    Observations.moving_average_daily_max,
)
Observations.filteredSeries = deprecate(
    'filteredSeries',
    Observations.filtered_series,
)

Simulations.fromTimeSeries = deprecate(
    'fromTimeSeries',
    Simulations.from_time_series,
)
Simulations.fromDataset = deprecate(
    'fromDataset',
    Simulations.from_dataset,
)
Simulations.subPeriod = deprecate(
    'subPeriod',
    Simulations.sub_period,
)
Simulations.dailyMean = deprecate(
    'dailyMean',
    Simulations.daily_mean,
)
Simulations.dailyMax = deprecate(
    'dailyMax',
    Simulations.daily_max,
)
Simulations.movingAverageDailyMax = deprecate(
    'movingAverageDailyMax',
    Simulations.moving_average_daily_max,
)
Simulations.filteredSeries = deprecate(
    'filteredSeries',
    Simulations.filtered_series,
)

Evaluator.subPeriod = deprecate(
    'subPeriod',
    Evaluator.sub_period,
)
Evaluator.stationSubList = deprecate(
    'stationSubList',
    Evaluator.station_sub_list,
)
Evaluator.subArea = deprecate(
    'subArea',
    Evaluator.sub_area,
)
Evaluator.selectCountries = deprecate(
    'selectCountries',
    Evaluator.select_countries,
)
Evaluator.quarterlyMedianScore = deprecate(
    'quarterlyMedianScore',
    Evaluator.quarterly_median_score,
)
Evaluator.meanTimeScores = deprecate(
    'meanTimeScores',
    Evaluator.mean_time_scores,
)
Evaluator.medianStationScores = deprecate(
    'medianStationScores',
    Evaluator.median_station_scores,
)
Evaluator.dailyMean = deprecate(
    'dailyMean',
    Evaluator.daily_mean,
)
Evaluator.dailyMax = deprecate(
    'dailyMax',
    Evaluator.daily_max,
)
Evaluator.movingAverageDailyMax = deprecate(
    'movingAverageDailyMax',
    Evaluator.moving_average_daily_max,
)
Evaluator.filteredSeries = deprecate(
    'filteredSeries',
    Evaluator.filtered_series,
)
Evaluator.stationScores = deprecate(
    'stationScores',
    Evaluator.temporal_scores,
)
Evaluator.timeScores = deprecate(
    'timeScores',
    Evaluator.spatial_scores,
)
Evaluator.FDscores = deprecate(
    'FDscores',
    Evaluator.spatiotemporal_scores,
)
Evaluator.FTscores = deprecate(
    'FTscores',
    Evaluator.temporal_ft_scores,
)
Evaluator.quarterlyMedianScore = deprecate(
    'quarterlyMedianScore',
    Evaluator.quarterly_score,
)
