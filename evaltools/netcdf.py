# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers netcdf processing functions."""

import glob
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np

import evaltools as evt

try:
    import netCDF4
except ImportError:
    print("!!! Python module netCDF4 is required to read netCDF files !!!")


def _simple_slice(arr, axis, starts, stops=None):
    """
    Slice a numpy array along axis, at starts/stops indices.

    Parameters
    ----------
    arr : numpy.array
        Array to slice.
    axis : list of integers
        List of axes where to perform the slices.
    starts : list of integers
        List of indices where to start slicing, corresponding to list of axes.
    stops : None or list of integers
        List of indices where to stop slicing, corresponding to list of axes.
        If None, returns array values at start index.

    Returns
    -------
    numpy.array
        Array (or value) sliced along the requested axes.

    """
    if stops is None:
        stops = [s + 1 for s in starts]

    sl = [slice(None)] * arr.ndim
    for ax, start, stop in zip(axis, starts, stops):
        if stop >= arr.shape[ax] - 1:
            sl[ax] = slice(start, None)
        elif stop == start + 1:
            sl[ax] = start  # no slice => will drop the axis
        else:
            sl[ax] = slice(start, stop)
    return arr[tuple(sl)]


def _slice(arr, index, stop=None):
    """Slice from 1D array/list with _simple_slice."""
    if stop is not None:
        stop = [stop]
    return _simple_slice(np.asarray(arr), [0], [index], stop)


def _grid_init(f, lon_name, lat_name, stations):
    """Initialise Interpolator class object with lon/lat grid."""
    nc = netCDF4.Dataset(f)
    lon = nc.variables[lon_name]
    lat = nc.variables[lat_name]
    coord_type = str(lon.ndim) + 'D'
    lon_0 = np.where(lon[:] > 180, lon[:]-360, lon[:])
    interp = evt.interpolation.Interpolator(lon_0, lat, coord_type)
    interp.load_stations_metadata(stations=stations)

    return interp


def _interpolate_netcdf(grid_obj, var):
    """
    Interpolate data from a netcdf file at all given stations.

    Parameters
    ----------
    grid_obj : interpolation.Interpolator object
        Object of class interpolation.Interpolator corresponding to current
        grid.
    var : netCDF4._netCDF4.Variable
        Variable from a netcdf file to interpolate, with dimensions
        time, lat and lon.

    Returns
    -------
    pandas.DataFrame
        DataFrame with no index and one column per interpolated station.

    """
    liste_series = []
    for t in range(var.shape[0]):
        liste_series.append(grid_obj.interpolate(var[t, :, :]))

    df = pd.concat(liste_series, axis=1)
    df = df.T

    return df


def _read_netcdf(f, start_time, end_time, stations, species,
                 series_type, date_format, level,
                 times_name, lon_name, lat_name, etape=False,
                 grid_obj=None, common_grid=True,
                 ign_keep=(0, 0), time_delta=None):
    """
    Collect data from netCDF files for specified stations.

    Parameters
    ----------
    f : str
        NetCDF file from which to read data
    start_time : datetime object
        The date from which data is collected.
    end_time : datetime object
        The date until which data is collected.
    stations : pandas.DataFrame
        DataFrame with station names as index, and metadata variables as
        columns.
    species : str
        Name of the variable to read in netCDF files.
    series_type : str
        Must be equal to 'hourly' if time series files contain one
        row per hour, or equal to 'daily' if they contain one row
        per day.
    date_format : str
        Format of dates contained in your netCDF files (strftime() reference).
        Special case : use '%Y%m%d.%f' for floats of type 20160925.25, and
        '%H' for times of type "hours/days since yyyy-mm-dd" OR if units and
        calendar attributes are set (automatic conversion).
    level : int or False
        Indicates which level (altitude) has to be taken. Use False if there
        are no levels in the files.
    times_name, lon_name, lat_name : str
        Names of the variables corresponding to dates/times, longitude
         and latitude.
    etape : boolean or "Stop"
        Used to stop reading of netCDF files if all dates have been found.
        If True, it will force to read data even if start_time has not been
        found.
        Modified to 'Stop' when end_date is found in current file.
    grid_obj : None or interpolation.Interpolator object
        If provided, grid_obj is recycled from file to file to spare computing
        time for stations interpolation. Must be None if files do not share the
        same grid !
    common_grid : boolean
        Indicates if all provided files share a common grid or not. If True,
        processing will be slightly faster.
    ign_keep : tuple of int
        For each set of datetimes in files, first int is number of hours/dates
        to ignore before fetching data. Second int is number of hours/dates of
        data to keep after that. e.g. if your files contain 96h of data but you
        only want values from second and third days (ie 48 hours from the 25th
        to the 72th hour), ign_keep should equal to (24, 48).
    time_delta : dict
        Dict of arguments passed to datetime.timedelta function, used to shift
        time values of netcdf file. Sometimes "cdo -showtimestamp" gives a
        different (and truest) times list than what may be computed based
        on "ncdump -v times".

    Returns
    -------
    pandas.DataFrame, str, interpolation.Grid[_unreg]
        DataFrame with datetime index build with all dates found in file f
        and one column per station ;
        string equals to 'Stop' if end_date has been found in file f ;
        grid is an object of class interpolation.Interpolator.

    """
    if grid_obj is None or common_grid is False:
        grid_obj = _grid_init(f, lon_name, lat_name, stations)

    nc = netCDF4.Dataset(f)

    times = nc.variables[times_name]
    times_type = times.dtype.kind 	# 'f' if float, 'S' if string
    # Get all datetimes of current file
    times_list = []
    # find right case to handle date conversion to python's datetime
    if times_type == 'S':  # string datetimes
        for x in times[:]:
            x = [str(s, 'utf-8') for s in x]  # for python3 compatibility
            if isinstance(x[0], str):
                times_list.append(datetime.strptime(''.join(x), date_format))
            elif times_list != []:  # empty date
                print("Problems encountered in Times variable of " + f)
                times_list.append(times_list[-1])
    else:  # float datetimes
        if date_format == "%H":
            try:
                times_list = netCDF4.num2date(times[:], times.units,
                                              times.calendar)
            except Exception:
                base_time = datetime.strptime(times.long_name.split()[-1],
                                              '%Y%m%d')
                times_list = [base_time + timedelta(hours=int(x))
                              for x in times[:]]
        elif date_format == "%Y%m%d.%f":
            times_list = [datetime.strptime(
                            str(x)[:8] + str(int(round(float(str(x)[8:])*24))),
                            '%Y%m%d%H')
                          for x in times[:]]
        else:
            times_list = [datetime.strptime(str(x), date_format)
                          for x in times[:]]

    # check which hours are to be kept
    t1 = 0
    t2 = len(times_list) - 1
    if ign_keep[0] > 0 and ign_keep[0] < len(times_list):
        t1 = ign_keep[0]
    if ign_keep[1] > 0 and t1+ign_keep[1] <= len(times_list):
        t2 = t1 + ign_keep[1]

    df = pd.DataFrame()
    if series_type == 'hourly':
        # prevent NaNs issued from non-full hours
        times_list = [(t + timedelta(minutes=29)).replace(
                        minute=0, second=0, microsecond=0) for t in times_list]
    elif series_type == 'daily':
        times_list = [t.date() for t in times_list]
    else:
        raise evt.EvaltoolsError("series_type argument must be 'hourly' or "
                                 "'daily' !!!")

    if time_delta is not None:
        times_list = [t + timedelta(**time_delta) for t in times_list]

    # check if we need to get data from file
    T1, T2 = 0, len(times_list) - 1
    if series_type == 'daily':
        # don't bother about filtering daily times (fast enough)
        etape = True
    elif (etape is False and
            times_list[t1] <= start_time <= _slice(times_list, t2)):
        print("Found first date in file " + f)
        etape = True
        T1 = times_list.index(start_time, t1, t2+1)
        if end_time in times_list:
            T2 = times_list.index(end_time, T1)
    elif times_list[t1] >= start_time and times_list[t1] <= end_time:
        etape = True
    T1 = max([T1, t1])
    T2 = min([T2, t2])
    if T2 == T1:
        T2 += 1

    if etape is True:
        dims = nc.variables[species].dimensions
        itimes, ilat, ilon, ilevel = 0, 1, 2, None

        # get var's dimensions in right order for interpolation
        for idim, dim in enumerate(dims):
            if dim in ['bottom_top', 'height', 'lev', 'level', 'zdim']:
                ilevel = idim
            elif dim in ['south_north', 'lat', 'latitude', 'y', lat_name]:
                ilat = idim
            elif dim in ['west_east', 'lon', 'longitude', 'x', lon_name]:
                ilon = idim
            elif dim in ['time', 'times', 'Time', 'Times',
                         'time_instant', 'time_counter', times_name]:
                itimes = idim
            else:
                print("Unable to recognise dimension {d}".format(d=dim))
                print("Trying to use dimensions : (time, lat, lon[, level])")

        var = _simple_slice(nc.variables[species][:], [itimes], [T1], [T2])
        if ilevel is not None:
            var = var.transpose(itimes, ilat, ilon, ilevel)
            var = var[:, :, :, level]
        else:
            var = var.transpose(itimes, ilat, ilon)

        df = _interpolate_netcdf(grid_obj, var)
        df.index = _slice(times_list, T1, T2)

    if end_time <= times_list[t2] and etape is True:
        print("Found last date in file "+f)
        etape = "Stop"

    return (df, etape, grid_obj)


def get_df(species, stations, generic_file_path, start_date, end_date, lag,
           correc_unit=1, series_type='hourly', date_format='%Y%m%d.%f',
           level=0, times_name="Times", lon_name="lon", lat_name="lat",
           common_grid=True, ign_keep=(0, 0), time_delta=None):
    """
    Collect data from netCDF files for specified stations.

    Parameters
    ----------
    species : str
        Name of the variable to read in netCDF files.
    stations : pandas.DataFrame
        DataFrame with station names as index, and metadata variables as
        columns.
    generic_file_path : str
        Generic path of netCDF files. * can be used as a joker character.
    start_date : datetime object
        The date from which data is collected.
    end_date : datetime object
        The data until which data is collected.
    lag : int
        Number of days with which the period start_date->end_date is shifted.
    correc_unit : float
        Multiplicative factor applied to original values.
    series_type : str
        Must be equal to 'hourly' if time series files contain one
        row per hour, or equal to 'daily' if they contain one row
        per day.
    date_format : str
        Format of dates contained in your netCDF files (strftime() reference).
        Special cases : use '%Y%m%d.%f' for floats of type 20160925.25, and
        '%H' for times of type "hours/days since yyyy-mm-dd" OR if units and
        calendar attributes are set (automatic conversion).
    level : int or False
        Indicates which level (altitude) has to be taken. Use False if there
        are no levels in the files.
    times_name, lon_name, lat_name : str
        Names of the variables corresponding to dates/times, longitude and
        latitude.
    common_grid : boolean
        Indicates if all provided files share a common grid or not. If True,
        processing will be slightly faster.
    ign_keep : tuple of int
        For each set of datetimes in files, first int is number of hours/dates
        to ignore before fetching data. Second int is number of hours/dates of
        data to keep after that. e.g. if your files contain 96h of data but you
        only want values from second and third days (ie 48 hours from the 25th
        to the 72th hour), ign_keep should equal to (24, 48).
    time_delta : dict
        Dict of arguments passed to datetime.timedelta function, used to shift
        time values of netcdf file. Sometimes "cdo -showtimestamp" gives a
        different (and truest) times list than what may be computed based
        on "ncdump -v times".

    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index and one column per station
        (stations which corresponding file is not found are dropped).

    """
    # Create dataframe with date[time] index
    start_date = start_date + timedelta(days=lag)
    end_date = end_date + timedelta(days=lag)
    nb_days = (end_date-start_date).days + 1
    if series_type == 'daily':
        times = [start_date+timedelta(days=d) for d in range(nb_days)]
        start_time = start_date
        end_time = end_date
    else:
        start_time = datetime.combine(start_date, time(0))
        end_time = datetime.combine(end_date, time(23))
        times = [start_time+timedelta(hours=h) for h in range(nb_days*24)]
    DF = pd.DataFrame(index=times,
                      columns=sorted(stations.index), dtype='float')

    # Find all matching patterns
    file_list = sorted(glob.glob(generic_file_path))
    if not file_list:
        print("Warning: no matching files found for path:")
        print(generic_file_path)

    # Loop on files to get all requested times-values
    etape = False
    grid_obj = None
    for f in file_list:

        # initialise Interpolator class object if all files use same grid
        if common_grid and file_list.index(f) == 0:
            print("Initialising grid information")
            grid_obj = _grid_init(f, lon_name, lat_name, stations)

        # read file, extract values at requested times and stations
        df, etape, grid_obj = _read_netcdf(
                                f,
                                start_time, end_time, stations,
                                species, series_type, date_format,
                                level, times_name, lon_name, lat_name,
                                etape, grid_obj, common_grid,
                                ign_keep, time_delta)

        # keep only requested hours
        t1 = 0
        t2 = df.shape[0]
        if ign_keep[0] > 0 and ign_keep[0] < df.shape[0]:
            t1 = ign_keep[0]
        if ign_keep[1] > 0 and t1+ign_keep[1] <= df.shape[0]:
            t2 = t1 + ign_keep[1]

        if not df.empty:
            DF.update(df[t1:t2], overwrite=True)

        if etape == "Stop":
            break

    if correc_unit != 1:
        DF = DF*correc_unit

    return DF


def simulations_from_netCDF(
        generic_file_path, stations,
        species, model, start, end, forecast_horizon=1,
        correc_unit=1, series_type='hourly',
        date_format='%Y%m%d.%f', level=0,
        availability_ratio=0.25, fill_value=None,
        times_name="Times", lon_name="lon", lat_name="lat",
        common_grid=True, nb_ignore=0, nb_keep=0,
        time_delta=None):
    """
    Construct a Simulations object from netcdf files.

    Multiple datetimes across files are overwritten.

    Parameters
    ----------
    generic_file_path : str
        Generic path of timeseriesfiles with {forecastDay} instead of
        the forecast day number.
    stations : pandas.DataFrame
        DataFrame with station names as index, and metadata variables as
        columns.
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
    date_format : str
        Format of dates contained in your netCDF files (strftime() reference).
        Special case : use '%Y%m%d.%f' for floats of type 20160925.25, and
        '%H' for times of type "hours/days since yyyy-mm-dd" OR if units and
        calendar attributes are set (automatic conversion).
    level : int or False
        Indicates which level (altitude) has to be taken. Use False if there
        are no levels in the files.
    availability_ratio : float or None or False
        Minimal rate of data available on the period required to
        keep a station. If None, stations with only nan values
        are dropped.
    fill_value : scalar or any valid FillValue from netCDF
        Indicates which value is used as NaN in netCDF files. Often equals
        -999.
    times_name, lon_name, lat_name : str
        Names of the variables corresponding to dates/times, longitude and
        latitude.
    common_grid : boolean
        Indicates if all provided files share a common grid or not. If True,
        processing will be slightly faster.
    nb_ignore : int
        For each set of datetimes in files, number of hours/dates to ignore
        before fetching data. e.g. if your files contain 96h of data but you
        only want values starting from second days (ie from the 25th hour),
        nb_ignore should equal to 24.
    nb_keep : int
        If nb_ignore is different from 0, nb_keep is used to indicate how many
        hours/dates of data should be kept, starting from nb_ignore. Any
        value <= 0 means all data from nb_ignore to end is kept.
    time_delta : dict
        Dict of arguments passed to datetime.timedelta function, used to shift
        time values of netcdf file. Sometimes "cdo -showtimestamp" gives a
        different (and truest) times list than what may be computed based
        on "ncdump -v times".

    Returns
    -------
    evaltools.evaluator.Simulations object.

    """
    obj = evt.evaluator.Simulations(start_date=start, end_date=end,
                                    stations=stations.index, species=species,
                                    model=model, series_type=series_type,
                                    forecast_horizon=forecast_horizon,
                                    path=generic_file_path)
    for fd in range(forecast_horizon):
        df = get_df(
                    species=species,
                    stations=stations,
                    generic_file_path=generic_file_path.format(
                                                            forecastDay=fd,
                                                            model=model,
                                                            polluant=species,
                                                            year=start.year),
                    start_date=start,
                    end_date=end,
                    lag=0,
                    correc_unit=correc_unit,
                    series_type=series_type,
                    date_format=date_format,
                    level=level, times_name=times_name,
                    lon_name=lon_name, lat_name=lat_name,
                    common_grid=common_grid,
                    ign_keep=(nb_ignore, nb_keep),
                    time_delta=time_delta)

        if fill_value is not None:
            df.replace(fill_value, np.nan, inplace=True)
        if not df.select_dtypes(include=['object']).empty:
            print("Warning : dataframe is of type 'object' instead of " +
                  "'float'. Trying to convert to float.")
            df = df.astype('float')
        obj.datasets[fd].updateFromDataset(df)
        print("Maximum value found in simulations Dataframe (" +
              str(fd) + ") : ", df.max().max())
    if availability_ratio is not False:
        obj.drop_unrepresentative_stations(availability_ratio)
    return obj
