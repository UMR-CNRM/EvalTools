# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module gathers ancillary functions."""

import numpy as np
import pandas as pd
from os.path import isfile
from sys import argv
import evaltools as evt


def _read_config_file(config_file):
    """Read configuration file with configparser."""
    import configparser
    conf = configparser.ConfigParser(inline_comment_prefixes='#')
    if isfile(config_file):
        conf.read(config_file)
        return conf
    else:
        raise evt.EvaltoolsError("{0} can't be found !!!".format(config_file))


def get_params():
    """
    Get console arguments and read configuration file.

    One of the console arguments must be: cfg=<configuration file path>.
    Configuration file must be structured like::

        [cat1]
        var1=...
        var2=...
        ...
        [cat2]
        ...

    The function will then returns the dictionary::

        {'console': {'cfg': <configuration file path>, 'arg1': ..., ...}
         'cat1': {'var1': ..., 'var2': ..., ...},
         'cat2': {'var1': ..., 'var2': ..., ...},
         ...}

    """
    params = {}
    consol_arg = {}
    for argument in argv[1:]:
        arg_name = argument.split('=')[0]
        arg_value = argument.split('=')[1]
        consol_arg[arg_name] = arg_value
    params['console'] = consol_arg

    conf = _read_config_file(consol_arg['cfg'])
    for sec in conf.sections():
        params[sec] = {}
        for k in conf[sec].keys():
            params[sec][k] = conf[sec][k]

    return params


def read_listing(
        listing_path, classes='all', species=None, types='all',
        area_coord=None, sub_list=None, sep=r'\s+',
        keep_all_cols=True, decimal='.', filters=None):
    """
    Read station list file.

    The input file must be a text file with one or mor spaces as data
    separator. The first row is interpreted as the header and must
    contain 'code', 'lat', 'lon' ('type', 'area' if a screening
    is performed on the station type, and 'class' or the species name
    if a screening is performed on the class). The first column must
    contain station codes.

    Parameters
    ----------
    listing_path : str
        Listing directory.
    classes : str
        If not 'all', stations are filtered by their class number. For
        example, specify classes="1-2-4" if you want to keep stations classed
        as 1, 2 or 4 only.
    species : str
        Only used if classes != 'all', it corresponds to the class column
        name in the listing.
    types : 'all' or list of tuple types
        For example types=[('bac','urb'), ('ind','urb'), ('tra','urb')].
        The first element of a tuple corresponds to the type column of the
        listing and the second one corresponds to the area column.
    area_coord : list
        List of the form [min_longitude, max_longitude, min_latitude,
        max_latitude] corresponding to the bounding box of the studied area.
        A station located exactly on the boundary will be accepted.
    sub_list : None or 1D array of str
        List of station to keep.
    sep : str
        Separator used in the listing file.
    keep_all_cols : bool
        If True, all columns of the listing files are kept in the returned
        dataframe. If False, only columns used for screening are kept.
    decimal : str
        Character to recognize as decimal point.
    filters : dict
        Dictionary with keys corresponding to one or several columns of the
        listing. The values of the dictionary are Boolean functions applied
        to the series corresponding to their key and rows with False result
        are discarded. The values can also be lists, in this case, the
        bolean function will be `x -> True if x is in the list else False`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing 'code', 'lat' and 'lon' of the
        filtered stations; 'type', 'area' if a screening is performed on the
        station type; and also their class if classes is not None.

    """
    if not isfile(listing_path):
        raise evt.EvaltoolsError("No station listing : {0} !!!".format(
            listing_path))

    if keep_all_cols is True:
        usecols = None
    else:
        usecols = ([u'code', u'lat', u'lon'] +
                   [u'type', u'area']*(types != 'all'))

    # filtering stations by class
    if classes != 'all':
        if species is not None:
            class_var = species
        else:
            class_var = u'class'
        if keep_all_cols is not True:
            usecols.append(class_var)
        lcla = classes.split('-')
        stations = pd.read_csv(listing_path, sep=sep, header=0,
                               index_col=0, usecols=usecols, decimal=decimal)
        stations[class_var] = stations[class_var].astype(str).str.strip()
        stations = stations[stations[class_var].isin(lcla)]
    else:
        stations = pd.read_csv(listing_path, sep=sep, header=0,
                               index_col=0, usecols=usecols, decimal=decimal)

    if stations.index.is_unique is False:
        raise evt.EvaltoolsError(
            "{} is corrupted : indexes are not unique !!!".format(listing_path)
        )

    # filtering stations by type
    if types != 'all':
        stations = stations[
            pd.Series(
                [(t, a) for t, a in zip(stations.type, stations.area)]
            ).isin(types).values
        ]

    if filters:
        for key, f in filters.items():
            if isinstance(f, list):
                def func(x):
                    return x in f
            else:
                func = f
            stations = stations.loc[[func(x) for x in stations[key]]]

    # filtering stations by area
    if area_coord is not None:
        idx_area = np.logical_and(
            np.logical_and(float(area_coord[0]) <= stations.lon,
                           float(area_coord[1]) >= stations.lon),
            np.logical_and(float(area_coord[2]) <= stations.lat,
                           float(area_coord[3]) >= stations.lat))
        stations = stations[idx_area]

    # keeping only a given sub list
    if sub_list is not None:
        idx = np.in1d(stations.index, sub_list)
        stations = stations.loc[idx]

    return stations
