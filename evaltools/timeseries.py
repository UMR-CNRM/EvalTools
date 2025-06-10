# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers time series processing functions."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from os.path import isfile
import sys
import warnings

import evaltools as evt
from evaltools._deprecate import deprecate
from evaltools._deprecate import deprecate_kwarg


@deprecate_kwarg('seriesType', 'series_type')
def readtimeseries(start_date, end_date, lag, lfiles, correc_unit=1,
                   series_type='hourly', keep='last', step=1):
    """
    Collect data from timeseries files.

    Timeseries files are text files where each lines contains a time
    (yyyymmddhh) or a date (yyyymmdd) and a value separated by spaces.

    Parameters
    ----------
    start_date : datetime object
        The date from which data is collected.
    end_date : datetime object
        The data until which data is collected.
    lag : int
        Number of days with which the period start_date->end_date is shifted.
    lfiles : list of str
        List containing the paths of files to read.
    correc_unit : float
        Multiplicative factor applied to original values.
    series_type : str
        Must be equal to 'hourly' if time series files contain one
        row per hour, or equal to 'daily' if they contain one row
        per day.
    keep : {'first', 'last', False}
        Adopted behavior when duplicated times are found in a file.
        - 'first' : keep the first occurrence only.
        - 'last' : keep the last occurrence only.
        - False : drop every occurences.
    step : int
        Time step in hours (ignored if series_type == 'daily').

    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index and one column containing
        read values.

    """
    start_date = start_date + timedelta(days=lag)
    end_date = end_date + timedelta(days=lag)

    if series_type == 'hourly':
        freq = timedelta(hours=step)
    elif series_type == 'daily':
        freq = timedelta(days=1)
    else:
        raise evt.EvaltoolsError(
            "series_type argument must be either 'hourly' or 'daily' !!!"
        )
    times = pd.date_range(
        start=start_date,
        end=datetime.combine(end_date, time(23)),
        freq=freq)

    # times_df = pd.DataFrame(index=times)
    df = pd.DataFrame(index=times)
    df['val'] = np.nan
    for f_path in lfiles:
        f = pd.read_csv(f_path, sep=r'\s+', names=['val'], index_col=0,
                        comment='#', dtype={'val': np.float64})
        f.index = f.index.astype(str)
        if series_type == 'hourly':
            f.index = pd.to_datetime(f.index, format='%Y%m%d%H')
            f = f.loc[np.logical_and(f.index >= times[0],
                                     f.index <= times[-1])]
        elif series_type == 'daily':
            f.index = pd.to_datetime(f.index, format='%Y%m%d').date
            f = f.loc[np.logical_and(f.index >= times[0].date(),
                                     f.index <= times[-1].date())]

        if f.index.is_unique is False:
            uniques = np.unique(np.array(f.index))
            # test if duplicated indexes have same values
            if len(uniques) != np.sum(~f.reset_index().duplicated()):
                print(("!!! {0} is corrupted : duplicated times with " +
                       "different values!!!\n").format(f_path))
            idx = ~f.index.duplicated(keep=keep)
            f = f.loc[idx]

        # df_temp = pd.merge(times_df, f, how='left', left_index=True,
        #                    right_index=True, sort=False)
        # df.update(df_temp)
        df.update(f)

    df.val = df.val*correc_unit
    df.loc[df.val < 0, 'val'] = np.nan

    return df


def readbigtimeseries(start_date, end_date, lag, fichier, series_type,
                      stations, correc_unit=1):
    """
    Collect data from big timeseries files.

    Big timeseries files are text files where each lines contains a station
    code, a time (yyyy-mm-dd_hh) and a value (separated by a space).

    Parameters
    ----------
    start_date : datetime object
        The date from which data is collected.
    end_date : datetime object
        The data until which data is collected.
    lag : int
        Number of days with which the period start_date->end_date is shifted.
    fichier : str
        Path to the file to read.
    correc_unit : float
        Multiplicative factor applied to original values.
    series_type : str
        Must be equal to 'hourly' if time series files contain one
        row per hour, or equal to 'daily' if they contain one row
        per day.
    stations : list of str
        Stations to keep.

    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index and one column containing
        read values.

    """
    start_date = start_date + timedelta(days=lag)
    end_date = end_date + timedelta(days=lag)

    if series_type == 'hourly':
        start_date = datetime.combine(start_date, time(0))
        end_date = datetime.combine(end_date, time(23))
    else:
        raise evt.EvaltoolsError(
            "series_type argument must be 'hourly' to "
            "use readbigtimeseries function !!!"
        )

    file_data = pd.read_csv(
        fichier,
        sep=' |_',
        names=['station', 'date', 'hour', 'val'],
        index_col=False,
        comment='#',
        dtype={'val': np.float64},
        engine='python')
    file_data = file_data.groupby(['station', 'date', 'hour'],
                                  as_index=False).mean()
    file_data.index = (pd.to_datetime(file_data.date, format="%Y-%m-%d") +
                       pd.to_timedelta(file_data.hour, unit='h'))
    file_data = file_data.loc[np.logical_and(file_data.index >= start_date,
                                             file_data.index <= end_date)]
    file_data = file_data.pivot(index=None, columns='station', values='val')
    stations_kept = [station for station in file_data.columns
                     if station in stations]
    stations_to_drop = [station for station in file_data.columns
                        if station not in stations]
    if len(stations_kept) <= 0:
        print(("Warning: none of the requested stations were found in " +
               "{fichier}\nStations to keep: {stations_kept}" +
               "\nStations to drop: {stations_to_drop}").format(
                    fichier=fichier,
                    stations_kept=stations_kept,
                    stations_to_drop=stations_to_drop))
    file_data.drop(columns=stations_to_drop, inplace=True, errors='ignore')

    return file_data


@deprecate_kwarg('stationsIdx', 'stations_idx')
def get_df(stations_idx, generic_file_path, start_date, end_date, lag,
           correc_unit=1, series_type='hourly', keep='last', step=1):
    """
    Collect data from timeseries files for specified stations.

    Parameters
    ----------
    stations_idx : list of str
        List containing the names of studied stations.
    generic_file_path : str
        Generic path of timeseriesfiles with {year} instead of the year number
        and {station} instead of the station name.
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
    keep : {'first', 'last', False}
        Adopted behavior when duplicated times are found in a file.
        - 'first' : keep the first occurrence only.
        - 'last' : keep the last occurrence only.
        - False : drop every occurences.
    step : int
        Time step in hours (ignored if series_type == 'daily').

    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index and one column per station
        (stations which corresponding file is not found are dropped).

    """
    df = pd.DataFrame()
    no_file_found = True
    for sta in stations_idx:
        # list of files to read
        file_list = [generic_file_path.format(year=year, station=sta)
                     for year in range(start_date.year, end_date.year+1)]

        # in case data from different years is kept in the same file
        file_list = np.unique(file_list)

        # files existence test
        idx = np.array([isfile(f) for f in file_list])
        file_list = file_list[idx]

        if idx.any():
            no_file_found = False
            new_df = readtimeseries(
                start_date=start_date,
                end_date=end_date,
                lag=lag,
                lfiles=file_list,
                correc_unit=correc_unit,
                series_type=series_type,
                keep=keep,
                step=step,
            )
            df[sta] = new_df['val']

    # if no files are found we still return an empty df with time index
    if df.empty:
        start_date = start_date + timedelta(days=lag)
        end_date = end_date + timedelta(days=lag)
        nb_days = (end_date-start_date).days + 1
        start_time = datetime.combine(start_date, time(0))
        times = [start_time+timedelta(hours=h) for h in range(nb_days*24)]
        df = pd.DataFrame(index=times)

    if no_file_found:
        print(
            f"Warning: no file found by timeseries.get_df "
            f"in {generic_file_path}"
        )

    return df


get_DF = deprecate(
    'get_DF',
    get_df,
)


def daily_mean(df, availability_ratio=0.75):
    """
    Compute daily mean.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns corresponding to stations and with datetime
        index in such way that there are 24 rows for each day.
    availability_ratio : float
        Minimal rate of data available in a day required
        to compute the daily mean.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns corresponding to stations and with
        date index.

    """
    # find number of values per day
    unique, counts = np.unique(df.index.date, return_counts=True)
    nval = counts[0]

    # dim0 = day, dim1 = hour of the day, dim2 = station
    df3d = df.values.reshape(int(df.shape[0]/nval), nval, df.shape[1])

    if availability_ratio >= 1.:
        print("*** Warning: availability_ratio of type float >= 1")
    availability_ratio = availability_ratio*nval

    idx_enough_values = np.sum(~np.isnan(df3d), axis=1) >= availability_ratio
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        res = np.nanmean(df3d, axis=1)
    res[~idx_enough_values] = np.nan
    res = pd.DataFrame(res, columns=df.columns,
                       index=np.unique(df.index.date))
    return res


def daily_max(df, availability_ratio=0.75):
    """
    Compute daily maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns corresponding to stations and with datetime
        index in such way that there are 24 rows for each day.
    availability_ratio : float
        Minimal rate of data available in a day required
        to compute the daily maximum.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns corresponding to stations and with
        date index.

    """
    # find number of values per day
    unique, counts = np.unique(df.index.date, return_counts=True)
    nval = counts[0]

    # dim0 = hour of the day, dim1=day, dim2=station
    df3d = df.values.reshape(int(df.shape[0]/nval), nval, df.shape[1])

    if availability_ratio >= 1.:
        print("*** Warning: availability_ratio of type float >= 1")
    availability_ratio = availability_ratio*nval

    idx_enough_values = np.sum(~np.isnan(df3d), axis=1) >= availability_ratio
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered')
        res = np.nanmax(df3d, axis=1)
    res[~idx_enough_values] = np.nan
    res = pd.DataFrame(res, columns=df.columns,
                       index=np.unique(df.index.date))
    return res


def moving_average(x, n, availability_ratio=0.75):
    """
    Compute the moving average with window size n on a vector x.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Vector used for computing.
    n : int
        Window size of the moving average.
    availability_ratio : float
        Minimal rate of values available to compute the average for a
        given window.

    Returns
    -------
        numpy.ndarray of length len(x)-n+1.

    """
    x = np.insert(x, 0, 0)
    mx = np.ma.masked_array(x, np.isnan(x))
    ret = np.cumsum(mx.filled(0))
    ret = ret[n:] - ret[:-n]
    counts = np.cumsum(~mx.mask)
    counts = counts[n:] - counts[:-n]
    with np.errstate(invalid='ignore'):
        res = ret/counts
    idx = counts >= n*availability_ratio
    res[~idx] = np.nan
    return res


def moving_average_daily_max(df, availability_ratio=0.75):
    """
    Compute the daily maximum eight hourly average.

    The maximum daily 8-hour mean concentration shall be selected by
    examining 8-hour running averages, calculated from hourly data and
    updated each hour. Each 8-hour average so calculated shall be assigned to
    the day on which it ends, that is, the first calculation period for any
    one day shall be the period from 17:00 on the previous day to 01:00 on
    that day; thelast calculation period for any one day will be the period
    from 16:00 to 24:00 on the day.

    As we need data from the day before to compute the daily maximum eight
    hourly average, the resulting DataFrame will contain one day less than
    the input one.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns corresponding to stations and with datetime
        index in such way that there are 24 rows for each day.
    availability_ratio : float
        Minimal rate of values available to compute the average for a given
        8-hour window, and also minimal rate of values available for a given
        day to compute the daily maximum of the moving average. For example
        with availability_ratio=0.75, a daily maximum eight hours average can
        only be calculated if 18 eight hours average are available each of
        which requires 6 hourly values to be available.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns corresponding to stations and with
        date index.

    """
    df_av = np.apply_along_axis(moving_average, 0, df.values[17:, :], n=8,
                                availability_ratio=availability_ratio)

    # dim0 = day, dim1 = hour of the day, dim2 = station
    df3d = df_av.reshape(int(df_av.shape[0]/24), 24, df_av.shape[1])

    idx_enough_values = (np.sum(~np.isnan(df3d), axis=1) >=
                         availability_ratio*24)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered')
        res = np.nanmax(df3d, axis=1)
    res[~idx_enough_values] = np.nan
    res = pd.DataFrame(res,
                       columns=df.columns,
                       index=np.unique(df.index.date)[1:])
    return res


def filtered_series(df, availability_ratio=0.75):
    """
    Substract daily cycle to the original series.

    The daily cycle is defined as the mean along all days for a given hour.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns corresponding to stations and with datetime
        index in such way that there are 24 rows for each day.
    availability_ratio : float
        Minimal rate of values available for a given hour to compute the
        daily filtered series for this hour in all days.

    Returns
    -------
        pandas.DataFrame with same shape as df.

    """
    # find number of values per day
    unique, counts = np.unique(df.index.date, return_counts=True)
    nval = counts[0]

    # dim0 = day, dim1 = hour of the day, dim2 = station
    arr3d = df.values.reshape(int(df.shape[0]/nval), nval, df.shape[1])
    hourly_count = (~np.isnan(arr3d)).sum(axis=0)
    hourly_sum = np.nansum(arr3d, axis=0)
    idx_enough_values = hourly_count >= (availability_ratio*nval)
    with np.errstate(invalid='ignore'):
        cycle = hourly_sum/hourly_count
    cycle[~idx_enough_values] = np.nan
    res = arr3d - cycle[np.newaxis, :, :]
    res = res.reshape(df.shape[0], df.shape[1])
    res = pd.DataFrame(res, columns=df.columns, index=df.index)
    return res


def normalized_series(df):
    """
    Normalize series by substracting the median and dividing by Q3-Q1.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns corresponding to stations and with datetime
        index.

    Returns
    -------
        pandas.DataFrame with same shape as df.

    """
    quartiles = pd.DataFrame(np.nanpercentile(df, q=[25, 50, 75], axis=0),
                             index=['q1', 'q2', 'q3'], columns=df.columns)
    quartiles.loc['iqr'] = quartiles.loc['q3'] - quartiles.loc['q2']
    res = (df - quartiles.loc['q2'])/quartiles.loc['iqr']
    return res


def check_timeseries_integrity(file_path, verbose=False, correction=False):
    """
    Check integrity of a timeseries file.

    Parameters
    ----------
    file_path : str
        Path of the file to read.
    verbose : bool
        If set to True, info is printed about the ts.
    correction : bool
        If set to True, the ts is corrected and the original file is
        overwritten.

    Returns
    -------
    ok : bool
        Boolean set to True if the timeseries is in a correct format or set
        to False if not.
    f : pandas.DataFrame
        Corrected timeseries

    """
    # redirecting printing output to /dev/null is verbose mode is off
    if verbose is True:
        print_output = sys.stdout
    else:
        print_output = open("/dev/null", "w")

    # boolean telling wether the timeseries is correct
    ok = True

    # bolean telling wether the timeseries is correctable
    correctable = True

    # reading timeseries
    f = pd.read_csv(file_path, sep=r'\s+', names=['val'], index_col=0)
    f.index = f.index.astype(str)
    f.index = pd.to_datetime(f.index, format='%Y%m%d%H')
    f.val[f.val < 0] = np.nan

    print("*** {} ***".format(file_path), file=print_output)

    # index without duplicates
    idx_unique = f.index.drop_duplicates()

    # checking for duplicates
    if f.index.is_unique is False:
        ok = False
        if len(idx_unique) != np.sum(~f.reset_index().duplicated()):
            print("- duplicated indexes with different values",
                  file=print_output)
            correctable = False
        else:
            print("- indexes are not unique (but duplicates have " +
                  "same values)", file=print_output)

    # checking for monotony
    if f.index.is_monotonic_increasing is False:
        ok = False
        print("- indexes are not sorted", file=print_output)

    # checking for missing indexes between min and max times of the timeseries
    sorted_idx = f.index.sort_values()
    start_time = sorted_idx[0]
    end_time = sorted_idx[-1]
    times = [start_time+timedelta(hours=h) for h in range(len(idx_unique))]
    if end_time != times[-1]:
        ok = False
        print("- missing indexes", file=print_output)

    # checking if first and last days are whole
    if start_time.hour != 0:
        ok = False
        print("- fisrt day begins at hour : {}".format(start_time.hour))
    if end_time.hour != 23:
        ok = False
        print("- last day ends at hour : {}".format(start_time.hour))

    # correcting timeseries is correction mode is on
    if correction is True and ok is False:
        if correctable is False:
            raise evt.EvaltoolsError(
                "*** duplicated indexes with "
                "different values !!! wich to choose?!"
            )
        nb_days = (end_time.date()-start_time.date()).days + 1
        times = [start_time+timedelta(hours=h) for h in range(nb_days*24)]
        f_correct = f.iloc[~f.index.duplicated()]
        f_correct = pd.merge(pd.DataFrame(index=times), f_correct, how='left',
                             left_index=True, right_index=True, sort=False)
        f_correct.to_csv(file_path, sep=' ', na_rep='nan',
                         float_format='%.12g', header=False, index=True,
                         date_format='%Y%m%d%H')
        f = f_correct

    if ok is True or correction is True:
        print("Summary :\
             \n- start : {start_date}\
             \n- end   : {end_date}\
             \n- nan   : {nb_nan}".format(start_date=start_time.date(),
                                          end_date=end_time.date(),
                                          nb_nan=np.sum(f.val.isna())),
              file=print_output)

    return ok, f


def check_timeseries_equality(file1, file2, start_date=None, end_date=None,
                              significant=5, verbose=False):
    """
    Check integrity of a timeseries file.

    Parameters
    ----------
    file1 : str
        Path of the first file to read.
    file2 : str
        Path of the second file to read.
    end_date : DateTime object
        The date when to end comparing the timeseries. If None, the maximum
        of the timeseries ending date is taken.
    significant : int
        Number of significant digits to take into account when comparing to
        values.
    start_date : DateTime object
        The date when to start comparing the timeseries. If None, the maximum
        of the timeseries starting date is taken.
    verbose : bool
        If set to True, info is printed when a difference is found.

    Returns
    -------
    bool
        Boolean set to True if the timeseries are equal or set to False
        if not.

    """
    # reading timeseries and checking for integrity
    b1, f1 = check_timeseries_integrity(file1)
    b2, f2 = check_timeseries_integrity(file2)

    if b1 is False or b2 is False:
        print(
            "Timeseries files are not correct! you can try "
            "check_timeseries_integrity(file_path, correction=True) to "
            "correct them."
        )
        return None

    # setting period
    if start_date is None:
        start_date = max(f1.index[0].date(), f2.index[0].date())
    if end_date is None:
        end_date = min(f1.index[-1].date(), f2.index[-1].date())
    start_time = datetime.combine(start_date, time(0))
    end_time = datetime.combine(end_date, time(23))
    f1 = f1.loc[np.logical_and(f1.index >= start_time, f1.index <= end_time)]
    f2 = f2.loc[np.logical_and(f2.index >= start_time, f2.index <= end_time)]

    # comparing values
    nb_days = (end_date-start_date).days + 1
    times = [start_time+timedelta(hours=h) for h in range(nb_days*24)]
    idx = []
    for t in times:
        try:
            np.testing.assert_approx_equal(
                f1.loc[t], f2.loc[t], significant=significant, verbose=False)
            idx.append(True)
        except AssertionError:
            idx.append(False)
            if verbose is True:
                print("for time {2} : {0} != {1}".format(f1.val.loc[t],
                                                         f2.val.loc[t],
                                                         t))

    if all(idx) is True:
        return True
    else:
        print(f"{len(idx)-sum(idx)} differences over {len(idx)} tested values")
        return False
