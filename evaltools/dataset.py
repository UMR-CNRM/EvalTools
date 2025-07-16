# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
This module defines Dataset and Store classes.

Dataset is designed to process input data from several formats.
Store is designed to store time series values in netcdf format.

"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import os
import copy
from packaging import version
import cftime
try:
    import netCDF4
except ImportError:
    pass

import evaltools as evt


class Dataset(object):
    """
    Dataset class for evaltools input data.

    This class is based on pandas.DataFrame class. The main attribute
    (data) is a pandas DataFrame with datetime index and a list of
    stations as columns.

    """

    def __init__(self, stations, start_date, end_date, species="",
                 series_type='hourly', step=1):
        """
        Dataset constructor.

        Parameters
        ----------
        stations : 1D array of str
            List of stations to keep in the returned Dataset.
        start_date : datetime.date object
            The date from which data is collected.
        end_date : datetime.date object
            The date until which data is collected.
        species : str
            Species to process.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep)
        step : int
            Time step in hours (ignored if series_type == 'daily').

        """
        # check arguments integrity
        if start_date > end_date:
            raise evt.EvaltoolsError(
                f"start_date={start_date} must be <= end_date={end_date}"
            )
        if not pd.Series(stations).is_unique:
            raise evt.EvaltoolsError(
                "Trying to construct a Dataset with duplicated stations."
            )
        if step not in [None, 1, 2, 3, 4, 6, 8, 12]:
            raise evt.EvaltoolsError(
                "Step argument must be either 1, 2, 3, 4, 6, 8 or 12."
            )

        # initialize attributes
        self.species = species
        self.nb_days = (end_date-start_date).days + 1
        self.start_date = start_date
        self.end_date = end_date

        # initialize dataframe according to the series type
        self.series_type = series_type
        if series_type == 'hourly':
            self.freq = timedelta(hours=step)
            self.step = step
        elif series_type == 'daily':
            self.freq = timedelta(days=1)
            self.step = None
        else:
            raise evt.EvaltoolsError(
                "series_type argument must be either 'hourly' or 'daily' !!!")
        times = pd.date_range(
            start=start_date,
            end=datetime.combine(end_date, time(23)),
            freq=self.freq)

        self.data = pd.DataFrame(index=times, columns=stations, dtype=float)
        self._metadata = pd.DataFrame(index=stations)

    @property
    def metadata(self):
        """Get the metadata."""
        return self._metadata.loc[self.data.columns]

    @property
    def date_format(self):
        """Get the date format according the series type of the data."""
        return ('%Y%m%d%H' if self.series_type == 'hourly' else '%Y%m%d')

    def nan_rate(self):
        """Compute rate of missing values in the dataframe."""
        nan_rate = (
            (self.data.isna().sum().sum()) /
            ((self.data.shape[0]*self.data.shape[1])*1.)
        )
        return nan_rate

    def summary(self):
        """
        Print summary statistics on the data.

        Nan rate, minimum and maximum are computed for each station. Then,
        the minimum, maximum and median of each of these statistics are
        displayed.

        """
        nan_rate = (self.data.isna().sum()/self.data.shape[0]).quantile(
            [0, .5, 1])
        min_val = self.data.min().quantile([0, .5, 1])
        max_val = self.data.max().quantile([0, .5, 1])

        summary = pd.DataFrame({'nan rate': nan_rate,
                                'min value': min_val,
                                'max value': max_val})
        summary.index.name = 'quantiles'
        print(summary)

    def add_stations_metadata(self, listing_path, **kwargs):
        """
        Read station metadata.

        The first column of the listing file must contain the station codes.
        Metadata is saved in the attribute self.stations.

        Parameters
        ----------
        listing_path : str
            Path to the listing file containing metadata.
        **kwargs :
            These parameters (like 'sep', 'sub_list', ...) will be
            passed to evaltools.utils.read_listing().

        """
        stations = evt.utils.read_listing(listing_path, **kwargs)
        if not np.in1d(self.data.columns, stations.index).all():
            raise evt.EvaltoolsError(
                "Listing does not contain all stations of the Dataset !!!")
        self._metadata = stations.loc[self.data.columns]

    def sub_period(self, start_date, end_date):
        """
        Build a new Dataset object define on a shorter period.

        Parameters
        ----------
        start_date : datetime.date
            Starting date of the new object.
        end_date : datetime.date
            Ending date of the new object.

        Returns
        -------
            Dataset object.

        """
        # check args integrity
        if start_date < self.start_date:
            raise evt.EvaltoolsError("start_date must be >= {} !!!".format(
                self.start_date))
        if end_date > self.end_date:
            raise evt.EvaltoolsError("end_date must be <= {} !!!".format(
                self.end_date))

        ds = self.__class__(stations=self.data.columns, start_date=start_date,
                            end_date=end_date, species=self.species,
                            series_type=self.series_type, step=self.step)
        ds.data.update(self.data)

        return ds

    def drop_unrepresentative_stations(self, availability_ratio=0.75,
                                       drop=True):
        """
        List stations with a certain rate of missing values.

        Parameters
        ----------
        availability_ratio : float or None
            Minimal rate of data available on the period required to
            keep a station. If None, stations with only nan values
            are dropped.
        drop : bool
            If True, the dataset is modify inplace by dropping
            unrepresentative stations.

        Returns
        -------
            List of stations that do not fulfill the condition.

        """
        if availability_ratio is None:
            idx = ((~self.data.isna()).sum() == 0.)
        else:
            idx = (
                (~self.data.isna()).sum()/self.data.shape[0] <
                availability_ratio
            )
        dropped_sta = idx.index[idx]
        if drop is True:
            self.data.drop(columns=dropped_sta, inplace=True)
        return dropped_sta

    def update_from_time_series(self, generic_file_path, correc_unit=1):
        """
        Update nan values of the current object with timeseries files.

        Timeseries files are text files where each lines contains a time
        (yyyymmddhh) or a date (yyyymmdd) and a value separated by spaces.

        Parameters
        ----------
        generic_file_path : str
            Generic path of timeseriesfiles with {year} instead of the year
            number and {station} instead of the station name.
        correc_unit : float
            Multiplicative factor applied to original values.

        """
        df = evt.timeseries.get_df(
            stations_idx=self.data.columns,
            generic_file_path=generic_file_path,
            start_date=self.start_date,
            end_date=self.end_date,
            lag=0,
            correc_unit=correc_unit,
            series_type=self.series_type,
            step=self.step)
        self.data.update(df)

    def update_from_dataset(self, dataset, correc_unit=1):
        """
        Update Dataset with values from an other one.

        Update nan values with values from an other Dataset or
        from a pandas.DataFrame.

        Parameters
        ----------
        dataset : evaltools.dataset.Dataset or pandas.DataFrame
            Dataset where to take data.
        correc_unit : float
            Multiplicative factor applied to original values.

        """
        try:
            self.data.update(dataset.data*correc_unit)
        except AttributeError:
            self.data.update(dataset*correc_unit)

    def add_new_stations(self, stations):
        """
        Add new stations to the Dataset.

        Add new stations to the Dataset if they are not already
        present. Modify Dataset object in place.

        Parameters
        ----------
        stations : 1D array-like
            List of stations to add.

        """
        new_stations = np.unique(stations)
        idx = ~np.in1d(new_stations, self.data.columns)
        new_stations = new_stations[idx]
        self.data = pd.concat(
            [self.data, pd.DataFrame(index=self.data.index,
                                     columns=new_stations)],
            axis=1,
            sort=False)

    def to_txt(self, output_path):
        """
        Write one txt file per station.

        Parameters
        ----------
        output_path : str
            Path of the output file. The Path must contain {station}
            instead of the station code.

        """
        for sta in self.data.columns:
            self.data[sta].to_csv(
                output_path.format(station=sta),
                sep=' ',
                na_rep='nan',
                float_format='%g',
                header=False,
                index=True,
                date_format=self.date_format)

    def to_netcdf(self, file_path, var_name=None, group=None, dim_names={},
                  coord_var_names={}, metadata_variables=[], **kwargs):
        """
        Write data in netcdf format.

        Parameters
        ----------
        file_path : str
            Path of the output file. If the file does not already exist, it is
            created.
        var_name : str
            Name of the variable to create.
        group : None or str
            Netcdf group where to store within the netcdf file. If equal to
            None, the root group is used.
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        metadata_variables : list of str
            List of metadata variables of self.metadata to add in the netcdf.
        kwargs : dict
            Additional keyword arguments passed to
            Store.new_concentration_var().

        """
        if os.path.isfile(file_path):
            if group is not None:
                store = Store.new_group(
                    file_path,
                    stations=self.data.columns,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    group=group,
                    dim_names=dim_names,
                    coord_var_names=coord_var_names,
                    series_type=self.series_type)
            else:
                store = Store(
                    file_path,
                    group=group,
                    read_only=False,
                    dim_names=dim_names,
                    coord_var_names=coord_var_names,
                    series_type=self.series_type)
        else:
            store = Store.new_file(
                file_path,
                stations=self.data.columns,
                start_date=self.start_date,
                end_date=self.end_date,
                group=group,
                dim_names=dim_names,
                coord_var_names=coord_var_names,
                series_type=self.series_type)

        if var_name is None:
            var_name = self.species
        try:
            store.nc_group[var_name]
        except IndexError:
            store.new_concentration_var(var_name, **kwargs)
        store.update(name=var_name, dataset=self)

        for var in metadata_variables:
            if var in self.metadata:
                values = pd.Series(index=store.stations[:], name=var)
                values.update(self.metadata[var])
                store.add_metadata_var(var, values)

        store = None

    def check_threshold(self, threshold, drop=False, file_path=None):
        """
        Check if values exceed a threshold.

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
        idx = (self.data > threshold)
        nb_val_above = np.sum(idx.values)
        if nb_val_above > 0:
            print(("*** {nb} values above threshold in the dataset !!! " +
                   "values are {action}.").format(
                       nb=nb_val_above,
                       action='kept'*(not drop)+'set to np.nan'*drop))
            if file_path is not None:
                self.data.stack().loc[idx.stack()].to_csv(
                    file_path, sep=' ', na_rep='nan', float_format='%g',
                    header=False, index=True, date_format=self.date_format)
            if drop is True:
                self.data[idx] = np.nan
        else:
            print("0 value above {} in the dataset.".format(threshold))

    def nb_values_timeseries(self):
        """Plot number of not nan values."""
        nb_val = (~self.data.isna()).sum(axis=1)
        ax = nb_val.plot()
        return ax


def delta_tool_formatting(dataset_dict, output_dir):
    """
    Write csv compatible with the JRC-DeltaTool.

    Parameters
    ----------
    dataset_dict : dictionary of evt.dataset.Dataset
        Keys of the dictionary must be species known by DeltaTool.
    output_dir : str
        Directory where to write the files (one file by measurement site).

    """
    station_dataframes = {}
    for lab, ds in dataset_dict.items():
        for sta in ds.data.columns:
            if sta in station_dataframes.keys():
                station_dataframes[sta][lab] = pd.DataFrame(
                    ds.data[sta].astype(float))
            else:
                station_dataframes[sta] = pd.DataFrame(
                    index=ds.data.index,
                    columns=['year', 'month', 'day', 'hour', lab])
                station_dataframes[sta].year = ds.data.index.year
                station_dataframes[sta].month = ds.data.index.month
                station_dataframes[sta].day = ds.data.index.day
                station_dataframes[sta].hour = ds.data.index.hour
                station_dataframes[sta][lab] = pd.DataFrame(
                    ds.data[sta].astype(float)
                )
    for sta, df in station_dataframes.items():
        station_dataframes[sta].to_csv(
            os.path.join(output_dir, "{}.csv".format(sta)),
            sep=';',
            na_rep='-999',
            float_format='%.5g',
            header=True,
            index=False,
        )


class Store(object):
    """
    Tool designed for storing time series data in netcdf format.

    To be handle by this class, netcdf variables must be 2-dimensional:
    the first dimension corresponding to time and the second one to
    the diferent measurement sites.

    """

    default_dim_names = {
        'time': 'time',
        'station_id': 'station_id',
    }

    default_coord_var_names = {
        'time': 'time',
        'station_id': 'station_id',
    }

    def __init__(self, file_path, group=None, read_only=False,
                 dim_names={}, coord_var_names={}, series_type='hourly'):
        """
        Store constructor.

        Parameters
        ----------
        file_path : str
            Path of the netcdf file to read. The file must exist (use the
            classmethod newfile to create one).
        group : None or str
            Group to read within the netcdf file. If equal to None, the root
            group is read.
        read_only : bool
            If True, the netcdf file is open in read only mode ('r'), otherwise
            in appendable mode ('r+').
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).

        """
        # set default names for dimensions and variables
        self.dim_names = copy.copy(self.default_dim_names)
        self.dim_names.update(dim_names)
        self.coord_var_names = copy.copy(self.default_coord_var_names)
        self.coord_var_names.update(coord_var_names)

        # open file
        if read_only is True:
            self.nc_root_group = netCDF4.Dataset(file_path, 'r')
        else:
            self.nc_root_group = netCDF4.Dataset(file_path, 'r+')
        if group is not None:
            self.nc_group = self.nc_root_group[group]
        else:
            self.nc_group = self.nc_root_group

        # verify dimensions availability
        for key in self.dim_names.values():
            if key not in self.nc_group.dimensions.keys():
                raise evt.EvaltoolsError(
                    'Dimension {key} not found in {file}'.format(
                        key=key, file=self.nc_group))

        # verify variables availability
        for key in self.coord_var_names.values():
            if key not in self.nc_group.variables.keys():
                raise evt.EvaltoolsError(
                    'Variable {key} not found in {file}'.format(
                        key=key, file=self.nc_group))

        times = self.get_times()
        dates = times.date
        self.start_date = dates.min()
        self.end_date = dates.max()
        self.nb_days = int((self.end_date - self.start_date).days + 1)

        self.series_type = series_type

        if not pd.Series(self.get_station_ids()).is_unique:
            print(("Warning: netcdf file {} contains duplicated station " +
                   "codes.").format(self.nc_root_group.filepath()))

    def __del__(self):
        """Delete magic method."""
        self.nc_root_group.close()

    def __enter__(self):
        """Enter magic method."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit magic method."""
        self.nc_root_group.close()

    @property
    def nsta(self):
        """Get the number of stations in the netcdf file."""
        return self.nc_group.dimensions[self.dim_names['station_id']].size

    @property
    def ntimes(self):
        """Get the number of times in the netcdf file."""
        return self.nc_group.dimensions[self.dim_names['time']].size

    @property
    def times(self):
        """Get the sequence of times in the netcdf file."""
        return self.nc_group.variables[self.coord_var_names['time']]

    @property
    def latitudes(self):
        """Get the sequence of latitudes in the netcdf file."""
        return self.nc_group.variables[self.coord_var_names['lat']]

    @property
    def longitudes(self):
        """Get the sequence of longitudes in the netcdf file."""
        return self.nc_group.variables[self.coord_var_names['lon']]

    @property
    def stations(self):
        """Get the station list in the netcdf file."""
        return self.nc_group.variables[self.coord_var_names['station_id']]

    @classmethod
    def new_file(cls, file_path, stations, start_date, end_date, group=None,
                 dim_names={}, coord_var_names={}, series_type='hourly',
                 step=1):
        """
        Build a Store object by writing a new netcdf file.

        If the file already exists, it is truncated.

        Parameters
        ----------
        file_path : str
            Path of the file to create.
        stations : 1D array of str
            List of stations in the returned Store object.
        start_date : datetime.date
            Starting date of the file data.
        end_date : datetime.date
            Ending date of the file data.
        group : None or str
            Netcdf group where to store within the netcdf file. If equal to
            None, the root group is used.
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            New Store object.

        """
        root_group = netCDF4.Dataset(file_path, 'w')
        if group is not None:
            nc_group = root_group.createGroup(group)
        else:
            nc_group = root_group

        cls._init_dimensions(
            nc_group=nc_group,
            stations=stations,
            start_date=start_date,
            end_date=end_date,
            dim_names=dim_names,
            coord_var_names=coord_var_names,
            series_type=series_type,
            step=step)

        root_group.close()

        return cls(file_path, group=group, read_only=False,
                   dim_names=dim_names, coord_var_names=coord_var_names,
                   series_type=series_type)

    @classmethod
    def new_group(cls, file_path, stations, start_date, end_date, group,
                  dim_names={}, coord_var_names={}, series_type='hourly',
                  step=1):
        """
        Create a new group inside a netcdf file.

        Parameters
        ----------
        file_path : str
            Path of the file to create.
        stations : 1D array of str
            List of stations in the returned Store object.
        start_date : datetime.date
            Starting date of the file data.
        end_date : datetime.date
            Ending date of the file data.
        group : None or str
            Netcdf group where to store within the netcdf file. If equal to
            None, the root group is used.
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            New Store object.

        """
        root_group = netCDF4.Dataset(file_path, 'r+')
        try:
            nc_group = root_group[group]
        except (IndexError, KeyError):
            nc_group = root_group.createGroup(group)
            cls._init_dimensions(
                nc_group=nc_group,
                stations=stations,
                start_date=start_date,
                end_date=end_date,
                dim_names=dim_names,
                coord_var_names=coord_var_names,
                series_type=series_type,
                step=step)

        root_group.close()

        return cls(file_path, group=group, read_only=False,
                   dim_names=dim_names, coord_var_names=coord_var_names,
                   series_type=series_type)

    @classmethod
    def _init_dimensions(cls, nc_group, stations, start_date, end_date,
                         dim_names, coord_var_names, series_type,
                         step=1):
        """
        Build a Store object by writing a new netcdf file.

        If the file already exists, it is truncated.

        Parameters
        ----------
        nc_group : netCDF4.Dataset
            Netcdf group to be initialized.
        stations : 1D array of str
            List of stations in the returned Store object.
        start_date : datetime.date
            Starting date of the file data.
        end_date : datetime.date
            Ending date of the file data.
        dim_names : dict
            Use to specify dimension names of the netcdf file. Default names
            are {'time': 'time', 'station_id': 'station_id'}.
        coord_var_names : dict
            Use to specify coordinate variable names of the netcdf file.
            Default names are {'time': 'time', 'station_id': 'station_id'}.
        series_type : str
            It can be 'hourly' (values stored with a hourly timestep) or
            'daily' (values stored with a daily timestep).
        step : int
            Time step in hours (ignored if series_type == 'daily').

        """
        # set default names for dimensions and variables
        dims = copy.copy(cls.default_dim_names)
        dims.update(dim_names)
        coord_vars = copy.copy(cls.default_coord_var_names)
        coord_vars.update(coord_var_names)

        # construct datetime index
        times = time_range(start_date, end_date, series_type, step=step)

        # create variables
        # time
        nc_group.createDimension(dims["time"], size=len(times))
        var_time = nc_group.createVariable(
            coord_vars["time"],
            "i4",
            (dims["time"]))
        var_time.setncattr("units", "hours since 1970-01-01 00:00:00")
        var_time.setncattr("calendar", "standard")
        var_time[:] = netCDF4.date2num(
            pd.to_datetime(times).to_pydatetime(),
            var_time.units,
            var_time.calendar)
        # station_id
        nc_group.createDimension(dims["station_id"],
                                 size=len(stations))
        var_station = nc_group.createVariable(
            coord_vars["station_id"],
            str,
            (dims["station_id"]))
        var_station[:] = np.array(stations)

    def get_station_ids(self):
        """Return station codes found in the netcdf group."""
        stations = self.stations[:]
        if stations.ndim == 1:
            return stations
        else:
            return netCDF4.chartostring(self.stations[:])

    def get_times(self):
        """Return time stepd found in the netcdf group."""
        if version.parse(cftime.__version__) >= version.parse('1.1.0'):
            time_convert = netCDF4.num2date(
                self.times[:],
                self.times.units,
                self.times.calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        else:
            time_convert = netCDF4.num2date(
                self.times[:],
                self.times.units,
                self.times.calendar,
            )
        return pd.DatetimeIndex(time_convert).round('h')

    def new_concentration_var(self, name, attrs={}, zlib=False,
                              least_significant_digit=None, complevel=4):
        """
        Add a new concentration variable to the netcdf group.

        Parameters
        ----------
        name : str
            Name of the variable to create.
        attrs : dict
            Dictionary of attributes for the new variable, (keys corresponding
            to name of the attributes and values to values of the attributes).
        zlib : bool
            if True, data assigned to the Variable instance is compressed
            on disk.
        least_significant_digit : int
            If specified, variable data will be truncated (quantized). In
            conjunction with zlib=True this produces 'lossy', but
            significantly more efficient compression. For example, if
            least_significant_digit=1, data will be quantized using
            around(scaledata)/scale, where scale = 2*bits, and bits is
            determined so that a precision of 0.1 is retained (in this
            case bits=4).
        complevel : int
            Level of zlib compression to use (1 is the fastest, but poorest
            compression, 9 is the slowest but best compression).

        """
        var = self.nc_group.createVariable(
            name,
            "f4",
            (self.dim_names["time"], self.dim_names["station_id"]),
            zlib=zlib,
            least_significant_digit=least_significant_digit,
            complevel=complevel,
        )
        for key, value in attrs:
            var.setncattr(key, value)

    def add_metadata_var(self, name, values, attrs={}):
        """
        Add a new concentration variable to the netcdf group.

        Parameters
        ----------
        name : str
            Name of the variable to create.
        values : 1D array like
            Values of the metadata variable as a vector of size the number
            of stations stored in the netcdf group.
        attrs : dict
            Dictionary of attributes for the new variable, (keys corresponding
            to name of the attributes and values to values of the attributes).

        """
        if name not in self.nc_group.variables:
            var = self.nc_group.createVariable(
                name,
                np.array(values).dtype,
                (self.dim_names["station_id"]),
            )
            var[:] = np.array(values)
            for key, value in attrs.items():
                var.setncattr(key, value)

    def add_stations(self, new_stations):
        """
        Add stations to the netcdf group.

        New station codes must not already be present.

        Parameters
        ----------
        new_stations : list of str
            List of the codes of the new stations.

        """
        raise evt.EvaltoolsError("{} not implemented.".format(
            self.add_stations))

        # var_station_id = self.nc_group.variables[
        #     self.coord_var_names['station_id']]

        # if np.in1d(new_stations, var_station_id[:]).any():
        #     raise evt.EvaltoolsError(
        #         "Trying to add already existing stations !!!")

        # self.stations[nsta:] = np.array(new_stations)

    def get_dataset(self, name, dataset_name=None, start_date=None,
                    end_date=None, stations=None, metadata_var={},
                    series_type='hourly', step=1, keep_dup_sta='first'):
        """
        Get data contained within a variable of the netcdf group.

        Requested variable must be 2-dimensional: the first dimension
        corresponding to time and the second one to the diferent measurement
        sites.

        Parameters
        ----------
        name : str
            Name of the variable to retrieve.
        dataset_name : str
            Species name given to the return dataset.
        start_date : datetime.date object
            The date from which data is collected.
        end_date : datetime.date object
            The date until which data is collected.
        stations : None or list of str
            List of stations to keep in the returned dataset.
        metadata_var : dict
            Dictionary that define metadata variables to get from de netcdf
            file. Keys of the provided dictionary are variable names as
            found in the file, and its values are variable names used for
            the returned dataset. These metadata variables must have one
            dimension only, corresponding to the station codes.
        step : int
            Time step in hours (ignored if series_type == 'daily').
        drop_duplicates : bool
            If True, stations with several entries are dropped.
        keep_dup_sta : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicated stations:
                - 'first' : drop duplicates except for the first
                    occurrence.
                - 'last' : drop duplicates except for the last
                    occurrence.
                - False : drop all duplicates.

        Returns
        -------
            evaltools.dataset.Dataset

        """
        # set period
        if start_date is not None:
            start = start_date
        else:
            start = self.start_date
        if end_date is not None:
            end = end_date
        else:
            end = self.end_date

        # find indices of data to retrieve
        times = self.get_times()
        idx = np.logical_and(times.date >= start, times.date <= end)

        # get DataFrame
        if idx.any():
            df = self.nc_group.variables[name]
            if df.dimensions == \
                    (self.dim_names['station_id'], self.dim_names['time']):
                df = df[:].T
            df = pd.DataFrame(
                df[idx, :],
                index=times[idx],
                columns=self.get_station_ids(),
            )
        else:
            print(
                (
                    "Warning: searching for data from {} to {} while input " +
                    "netcdf file contain data from {} to {}."
                ).format(start, end, self.start_date, self.end_date)
            )
            df = pd.DataFrame(
                index=times[idx],
                columns=self.get_station_ids(),
            )

        dup_col = df.columns.duplicated(keep=keep_dup_sta)
        if dup_col.any():
            print(
                f"Warning: duplicated station found, dropping "
                f"{df.columns[dup_col]}."
            )
        df = df.loc[:, ~dup_col].copy()

        if stations is not None:
            df = df[stations]

        # construct Dataset
        if dataset_name is None:
            species = name
        else:
            species = dataset_name
        res = Dataset(stations=df.columns,
                      start_date=start,
                      end_date=end,
                      species=species,
                      series_type=series_type,
                      step=step)
        res.data.update(df)

        # get metadata
        for nc_name, ds_name in metadata_var.items():
            meta = pd.DataFrame(self.nc_group.variables[nc_name][~dup_col],
                                index=self.get_station_ids()[~dup_col],
                                columns=[ds_name])
            if stations is not None:
                meta = meta.loc[stations]
            res._metadata[ds_name] = np.nan
            res._metadata.update(meta)

        return res

    def update(self, name, dataset, add_new_stations=False):
        """
        Update a variable of the netcdf file with a Dataset object.

        Modify a variable of the netcdf file using non-NA values from passed
        Dataset object.

        Parameters
        ----------
        name : str
            Name of the variable to update.
        dataset : evaltools.dataset.Dataset object
            Dataset object containing new values to add to the file.
        add_new_stations : bool
            If true, stations only present in dataset are also
            added to the netcdf file.

        """
        # check if series types match
        if self.series_type != dataset.series_type:
            raise evt.EvaltoolsError(
                "Series types of dataset and self must be equal !!!")

        # check if netcdf file holds the appropriate variable
        if name not in self.nc_group.variables.keys():
            raise evt.EvaltoolsError(
                "'{var}'' variable not found in {store}.".format(
                    var=name, store=self))

        # get station list from netcdf file and add some if missing
        if add_new_stations:
            new_stations = dataset.data.columns[
                np.logical_not(np.in1d(dataset.data.columns,
                                       self.stations[:]))]
            self.add_stations(new_stations)

        # find indices of data to retrieve
        times = self.get_times()
        idx = np.logical_and(times.date >= dataset.start_date,
                             times.date <= dataset.end_date)

        # get DataFrame
        df = pd.DataFrame(self.nc_group.variables[name][idx],
                          index=times[idx],
                          columns=self.get_station_ids())

        # write updated dataset in the h5 file
        if dataset.data.shape == df.shape and \
                (df.index == dataset.data.index).all() and \
                (df.columns == dataset.data.columns).all():
            df = dataset.data
        else:
            df.update(dataset.data)
        self.nc_group.variables[name][idx] = np.ma.array(df.values)


def time_range(start_date, end_date, series_type, step=1):
    """
    Create a time range.

    Parameters
    ----------
    start_date : datetime.date object
        The date from which the range starts.
    end_date : datetime.date object
        Ending date (included) of the range.
    series_type : str.
        It can be 'hourly' (hourly timestep) or
        'daily' (daily timestep).
    step : int
        Time step in hours (ignored if series_type == 'daily').

    Returns
    -------
        List of datetime.date or datetime.datetime

    """
    if series_type == 'hourly':
        freq = timedelta(hours=step)
    elif series_type == 'daily':
        freq = timedelta(days=1)
    else:
        raise evt.EvaltoolsError(
            "series_type argument must be either 'hourly' or 'daily' !!!")
    times = pd.date_range(
        start=start_date,
        end=datetime.combine(end_date, time(23)),
        freq=freq)
    return times
