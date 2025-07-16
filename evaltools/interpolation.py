# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
This module defines Grid.

Grid is a class designed to interpolate values from a grid of values.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six import iteritems
from functools import reduce
from math import ceil, floor

import evaltools as evt


class Grid(object):
    """
    Class designed to interpolate data from a lat/lon grid.

    The grid must have equidistant latitudes/longitudes.

    """

    def __init__(self, min_lat, min_lon, d_lon, d_lat, nb_lon, nb_lat):
        """
        Class constructor.

        Parameters
        ----------
        min_lat : scalar
            Minimal latitude values of the grid.
        min_lon : scalar
            Minimal longitude values of the grid.
        d_lon : scalar
            Longitude step width of the grid.
        d_lat : scalar
            Latitude step width of the grid.
        nb_lon : int
            Number of longitude steps in the grid (shape[1] of the grid).
        nb_lat : int
            Number of latitude steps in the grid (shape[0] of the grid).

        """
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.d_lon = d_lon
        self.d_lat = d_lat
        self.nb_lon = nb_lon
        self.nb_lat = nb_lat
        self.max_lat = min_lat + d_lat*(nb_lat - 1)
        self.max_lon = min_lon + d_lon*(nb_lon - 1)
        self.grids = None
        self.stations = None
        self.nearest_points = None
        self.stations_to_interpolate = {}

    def load_stations_metadata(self, listing_path, **kwargs):
        """
        Load stations coordinate and find nearest grid points.

        Retrieve a list of stations with their coordinates and find
        the four nearest grid points from each of them for a later
        use in the interpolation method.

        The listing file is read with evaltools.utils.read_listing
        and therefore it must be in the correct format.

        Parameters
        ----------
        listing_path : str
            Path of the listing file.
        **kwargs :
            These parameters (like 'sep', 'sub_list', ...) will be
            passed to evaltools.utils.read_listing().

        """
        # read listing
        self.stations = evt.utils.read_listing(
            listing_path,
            area_coord=[
                self.min_lon, self.max_lon,
                self.min_lat, self.max_lat,
            ],
            **kwargs,
        )

        idx = reduce(np.logical_and, (self.stations.lon >= self.min_lon,
                                      self.stations.lon <= self.max_lon,
                                      self.stations.lat >= self.min_lat,
                                      self.stations.lat <= self.max_lat))
        dropped_stations = np.sum(~idx)
        if dropped_stations > 0:
            self.stations = self.stations.loc[idx]
            print(
                f"{dropped_stations} stations dropped since they are "
                f"located outside the domain."
            )

        # find four nearest grid points
        self.nearest_points = {}
        for sta in self.stations.index:
            self.nearest_points[sta] = find_nearest(
                obs_lon=self.stations.loc[sta].lon,
                obs_lat=self.stations.loc[sta].lat,
                max_lat=self.max_lat,
                min_lat=self.min_lat,
                max_lon=self.max_lon,
                min_lon=self.min_lon,
                d_lon=self.d_lon,
                d_lat=self.d_lat,
                nb_lon=self.nb_lon,
                invert_lat=True)

    def load_station_lists(self, station_lists):
        """
        Import station lists used for interpolation.

        Parameters
        ----------
        station_lists : dict of 1D arrays
            Dictionary which keys are names for the different grids (species,
            hours, ...) and values are 1D arrays of station codes.

        """
        # check if self.stations has been initialised
        if self.stations is None:
            print("*** self.stations is None !!! it must be initialised "
                  "with load_stations_metadata object method.")
            return None

        for key, val in iteritems(station_lists):
            idx_in_stations = np.in1d(np.array(val), self.stations.index)
            if np.sum(~idx_in_stations) != 0:
                print(
                    f"*** {sum(~idx_in_stations)} stations dropped since "
                    f"they do not appear in self.stations.index "
                    f"(these are stations not found in the metadata file "
                    f"or located outside the domain)."
                )
            self.stations_to_interpolate[key] = val[idx_in_stations]

    def set_grids(self, grids):
        """
        Set grids values from a dictionary of 2D arrays.

        Parameters
        ----------
        grids : dict of 2D arrays
            Dictionary which keys are are names for the different grids
            (species, hours, ...), and values are 2D arrays of float with
            shape corresponding to (self.nb_lat, self.nb_lon).

        """
        for key, grid in iteritems(grids):
            assert grid.shape == (self.nb_lat, self.nb_lon), \
                "grids[{key}] do not fit {nb_lat}x{nb_lon} shape".format(
                    key=key, nb_lat=self.nb_lat, nb_lon=self.nb_lon)
        self.grids = grids

    def view(self):
        """Visualize grid values."""
        if self.grids is not None:
            for key, im in iteritems(self.grids):
                plt.close('all')
                plt.title(key)
                plt.imshow(im)
                plt.axis('off')
                plt.show()
        else:
            print("*** self.grids is None !!!")

    def interpolate(self, use_same_listing=False):
        """
        Interpolate values for all loaded grids.

        Parameters
        ----------
        use_same_listing : bool
            If False, for each self.grids keys, a list of
            stations for interpolation must have been submitted with
            Grid.load_station_lists method.
            If True, interpolation is done at every stations
            found in the listing file fed to Grid.load_stations_metadata.
        Returns
        -------
        dictionary
            Dictionary with one key per grid, containing pandas.Series of
            interpolated values.

        """
        # check if grids and station lists have been initialised
        if self.grids is None:
            print("*** self.grids is None !!!")
            return None
        if use_same_listing is False:
            idx_loaded_lists = np.in1d(self.grids.keys(),
                                       self.stations_to_interpolate.keys())
            if idx_loaded_lists.all() is False:
                print(("*** No station list for grid {} !!! must be "
                       "initialised with load_station_lists object "
                       "method.").format(self.grids.keys()[~idx_loaded_lists]))
                return None
        else:
            # check if self.stations has been initialised
            if self.stations is None:
                print("*** self.stations is None !!! it must be initialised "
                      "with load_stations_metadata object method.")
                return None

        # interpolation
        res = {}
        for key, grid in iteritems(self.grids):
            field = grid.flatten()
            vals = {}
            if use_same_listing is False:
                coords_to_interpolate = self.stations.loc[
                    self.stations_to_interpolate[key]][['lat',
                                                        'lon']].itertuples()
            else:
                coords_to_interpolate = self.stations[['lat',
                                                       'lon']].itertuples()
            for station, inlat, inlon in coords_to_interpolate:
                points = self.nearest_points[station]
                if len(points['indices']) == 1:
                    vals[station] = field[int(points['indices'][0])]
                else:
                    p = [(lon, lat, field[int(idx)])
                         for idx, lat, lon in zip(points["indices"],
                                                  points["latitudes"],
                                                  points["longitudes"])]
                    vals[station] = bilinear_interpolation(
                        inlon, inlat, p, sort=False)
            res[key] = pd.Series(vals)
        return res


def find_nearest(
        obs_lat, obs_lon, max_lat, min_lat, max_lon, min_lon, d_lon, d_lat,
        nb_lon, invert_lat=True):
    """
    Find the 4 nearest grid points for a given lat/lon.

    The grid must have equidistant latitudes/longitudes.

    Returned indices are corresponding to grid data stored in
    a 1D-array (ie the grid is flatten in row-major (C-style) order).

    Parameters
    ----------
    obs_lon : float
        Longitude of the input observation.
    obs_lat : float
        Latitude of the input observation.
    max_lat : float
        Minimal latitude of the grid.
    min_lon : float
        Minimal longitude of the grid.
    d_lat : float
        Latitude step width of the grid.
    d_lon : float
        Longitude step width of the grid.
    nb_lon : int
        Number of longitude steps in the grid (shape[1] of the grid).
    invert_lat : bool
        Must be set to True if latitudes are stored in decreasing
        order in the grid.

    Returns
    -------
    dictionary
        Dictionary which keys are 'latitudes', 'longitudes' and 'indices',
        and values are list of four floats.

    """
    # verify args consistency
    max_lon = min_lon + d_lon*(nb_lon-1)
    if obs_lat > max_lat or obs_lon < min_lon or obs_lon > max_lon \
            or obs_lat < min_lat or obs_lon < min_lon:
        raise evt.EvaltoolsError('Input point outside the grid !!!')

    # find lat/lon indices of the upper left grid point of the mesh
    # containing the input point
    j = (1 - invert_lat*2)*(obs_lat - max_lat)/d_lat
    i = (obs_lon - min_lon) / d_lon

    if int(i) == i and int(j) == j:
        mod_idx = [i + j*nb_lon]
        mod_lat = [obs_lat]
        mod_lon = [obs_lon]
    else:
        i = int(i)
        j = int(j)
        # calculate min/max latitudes of the mesh containing the input point
        mod_lat_0 = max_lat + (1 - invert_lat * 2)*j*d_lat
        mod_lat_1 = mod_lat_0 + (1 - invert_lat * 2)*d_lat

        # calculate min/max longitudes of the mesh containing the input point
        mod_lon_0 = min_lon + i*d_lon
        mod_lon_1 = mod_lon_0 + d_lon

        # calc 1D indices of grid points of the mesh containing the input point
        mod_idx_0 = i + j*nb_lon
        mod_idx_1 = mod_idx_0 + 1
        mod_idx_2 = i + (j + 1)*nb_lon
        mod_idx_3 = mod_idx_2 + 1

        # result
        mod_idx = [mod_idx_2, mod_idx_0, mod_idx_3, mod_idx_1]
        mod_lat = np.round([mod_lat_1, mod_lat_0, mod_lat_1, mod_lat_0], 5)
        mod_lon = np.round([mod_lon_0, mod_lon_0, mod_lon_1, mod_lon_1], 5)

    result = {'latitudes': mod_lat, 'longitudes': mod_lon, 'indices': mod_idx}
    return result


class Interpolator(object):
    """Class designed to interpolate data from any kind of lat/lon grid."""

    def __init__(self, lon, lat, coord_type):
        """
        Class constructor.

        Parameters
        ----------
        lat : netCDF4._netCDF4.Variable
            Latitude values of the grid.
        lon : netCDF4._netCDF4.Variable
            Longitude values of the grid.
        coord_type : str
            Must be equal to '2D' if longitude and latitude are 2D variables,
            and equal to '1D' otherwise.

        """
        self.lat = np.array(lat).flatten()
        self.lon = np.array(lon).flatten()
        self.min_lat = self.lat.min()
        self.min_lon = self.lon.min()
        self.max_lat = self.lat.max()
        self.max_lon = self.lon.max()
        self.lonshape = lon.shape[0] if coord_type == '1D' else lon.shape[1]
        self.latshape = lat.shape[0]

        # self.coord_type = coord_type #2D or 1D
        self.ccoord = 1 if coord_type == '1D' else self.lonshape
        self.c0 = 0 if coord_type == '1D' else 1
        self.c1 = 1

        # Determine useful parameters for regular grid
        self.d_lon, self.d_lat = 0., 0.
        self.regular = False
        self._grid_type()

        # self.grids = None
        self.stations = None        # only contains stations inside domain
        self.nearest_point = None    # contains all stations
        self.nearest_points = None   # only contains stations inside domain

    def _grid_type(self):
        """Determine d_lon and d_lat if grid is regular."""
        cond = (
            (
                self.lon[self._flon(0, 0)] ==
                self.lon[self._flon(self.latshape-1, 0)]
            )
            and (
                self.lat[self._flat(0, 0)] ==
                self.lat[self._flat(0, self.lonshape-1)]
            )
        )
        if cond:
            d_lon = abs(
                abs(self.lon[self._flon(0, 1)]) -
                abs(self.lon[self._flon(0, 0)])
            )
            d_lat = abs(
                abs(self.lat[self._flat(1, 0)]) -
                abs(self.lat[self._flat(0, 0)])
            )
            self.d_lon, self.d_lat = d_lon, d_lat
            self.regular = True
            print("Grid is regular")

    def summary(self):
        """Print a summary of the object."""
        print(f"min_lat, min_lon : {self.min_lat}, {self.min_lon}")
        print(f"max_lat, max_lon : {self.max_lat}, {self.max_lon}")
        print(f"latshape, lonshape : {self.latshape}, {self.lonshape}")
        print(f"d_lat, d_lon : {self.d_lat}, {self.d_lon}")

    def load_stations_metadata(self, listing_path=None,
                               stations=None, **kwargs):
        """
        Load stations coordinate and find nearest grid points.

        Retrieve a list of stations with their coordinates and find
        the four nearest grid points from each of them for a later
        use in the interpolation method.

        The listing file is read with evaltools.utils.read_listing
        and therefore it must be in the correct format.

        Parameters
        ----------
        listing_path : str
            Path of the listing file.
        stations : pandas.DataFrame
            DataFrame with station names as index, and metadata variables as
            columns.
        **kwargs :
            Additional parameters (like 'sep', 'sub_list', ...) to be
            passed to evaltools.utils.read_listing().

        """
        print(self.summary())
        # read listing
        if listing_path is not None:
            stations = evt.utils.read_listing(listing_path)
        elif stations is not None:
            stations = stations
        else:
            raise evt.EvaltoolsError(
                'listing_path or stations must be provided ' +
                'to use interpolation.load_stations_metadata !!!')

        # find nearest grid point for each station
        self.nearest_point = {}
        for sta in stations.index:
            self.nearest_point[sta] = self.find_nearest(
                                            point=(stations.loc[sta].lon,
                                                   stations.loc[sta].lat))
        # drop stations not strictly inside domain
        self.stations = self.filter_stations(stations)
        self.nearest_points = {}
        for sta in self.stations.index:
            self.nearest_points[sta] = self.find_four_nearest(sta)

    def filter_stations(self, stations):
        """
        Drop stations not contained in domain.

        Check if nearest point is part of the domain's border, and drop
        station if true.

        Parameters
        ----------
        stations : pandas.DataFrame
            DataFrame with station names as index, and metadata variables
            as columns.
        """
        sta_to_drop = []
        for sta in stations.index:
            if self.nearest_point[sta]['ilat'] in [-1, 0, self.latshape-1] or \
                    self.nearest_point[sta]['jlon'] in \
                    [-1, 0, self.lonshape-1]:
                sta_to_drop.append(sta)

        return stations.drop(sta_to_drop)

    def find_nearest(self, point):
        """
        Find nearest grid point.

        Parameters
        ----------
        point : tuple of two floats
            Coordinates (lon, lat) of the input point.

        """
        dist = 10000.
        ilat, jlon = -1, -1

        if self.regular:
            # find nearest grid point, using d_lat and d_lon
            if point[1] > self.max_lat or point[1] < self.min_lat:
                ii = [0]*2
            else:
                ii = [
                    int(min([
                        max(
                            [
                                self.latshape - 1 -
                                ceil((self.max_lat - point[1])/self.d_lat),
                                0,
                            ]
                        ),
                        self.latshape - 1,
                    ])),
                    int(min([
                        max(
                            [floor((point[1] - self.min_lat)/self.d_lat), 0]
                        ),
                        self.latshape - 1,
                    ]))
                ]

            if point[0] < self.min_lon or point[0] > self.max_lon:
                jj = [0]*2
            else:
                jj = [
                    int(min([
                        max(
                            [
                                self.lonshape - 1 -
                                ceil((self.max_lon - point[0])/self.d_lon),
                                0,
                            ]
                        ),
                        self.lonshape - 1,
                    ])),
                    int(min([
                        max(
                            [floor((point[0] - self.min_lon)/self.d_lon), 0]
                        ),
                        self.lonshape - 1,
                    ]))
                ]

            for i, j in zip(ii, jj):
                if self.lat[0] > self.lat[-1]:
                    i = self.latshape - 1 - i
                d = squared_dist(
                    point,
                    (self.lon[self._flon(i, j)],
                     self.lat[self._flat(i, j)])
                )
                if d < dist:
                    dist = d
                    ilat, jlon = i, j

        else:
            # check distance between point and all grid points
            # (to be replaced by a more efficient method)
            for i in range(self.latshape):
                for j in range(self.lonshape):
                    d = squared_dist(
                        point,
                        (self.lon[self._flon(i, j)],
                         self.lat[self._flat(i, j)])
                    )
                    if d < dist:
                        dist = d
                        ilat, jlon = i, j

        return {'ilat': ilat, 'jlon': jlon,
                'flat_lat': self._flat(ilat, jlon),
                'flat_lon': self._flon(ilat, jlon)}

    def find_four_nearest(self, station):
        """
        Find all four nearest grid points of station.

        Parameters
        ----------
        station : str
            Station name. Must be in self.nearest_point.

        """
        if self.nearest_point is None:
            print("*** self.nearest_point is None !!! it must be initialised "
                  "with load_stations_metadata object method.")
            return None

        # get information about nearest point of station
        iref, jref, flatref, flonref = [
            self.nearest_point[station][key]
            for key in ['ilat', 'jlon', 'flat_lat', 'flat_lon']
        ]
        latref, lonref = self.stations.loc[station][['lat', 'lon']]
        # define outputs
        ilats = [iref, -1, -1, -1]
        jlons = [jref, -1, -1, -1]
        flats = [flatref, -1, -1, -1]
        flons = [flonref, -1, -1, -1]
        # find nearest points
        # first, longitude direction (direct because array-direction)
        d1 = squared_dist(
            (lonref, latref),
            (self.lon[self._flon(iref, jref+1)],
             self.lat[self._flat(iref, jref+1)])
        )
        d2 = squared_dist(
            (lonref, latref),
            (self.lon[self._flon(iref, jref-1)],
             self.lat[self._flat(iref, jref-1)])
        )
        id_lon = 1 if d1 < d2 else -1
        ilats[1] = iref
        jlons[1] = jref+id_lon
        flats[1] = self._flat(iref, jref+id_lon)
        flons[1] = self._flon(iref, jref+id_lon)

        # second, latitude direction (a bit more tricky because of flat arrays)
        d1 = squared_dist(
            (lonref, latref),
            (self.lon[self._flon(iref+1, jref)],
             self.lat[self._flat(iref+1, jref)])
        )
        d2 = squared_dist(
            (lonref, latref),
            (self.lon[self._flon(iref-1, jref)],
             self.lat[self._flat(iref-1, jref)])
        )
        id_lat = 1 if d1 < d2 else -1
        ilats[3] = iref+id_lat
        jlons[3] = jref
        flats[3] = self._flat(iref+id_lat, jref)
        flons[3] = self._flon(iref+id_lat, jref)

        # get last point, close the square
        ilats[2] = iref+id_lat
        jlons[2] = jref+id_lon
        flats[2] = self._flat(iref+id_lat, jref+id_lon)
        flons[2] = self._flon(iref+id_lat, jref+id_lon)

        result = {'latitudes': [self.lat[f] for f in flats],
                  'longitudes': [self.lon[f] for f in flons],
                  'indices': {'ilat': ilats,
                              'jlon': jlons}}
        return result

    def interpolate(self, values):
        """
        Interpolate values on lon/lat grid at provided stations locations.

        Parameters
        ----------
        values : 2D array
            Grid of values, corresponding to lon/lat grid.

        Returns
        -------
        pandas.Series
            pandas.Series of interpolated values.

        """
        if self.stations is None:
            print("*** self.stations is None !!! it must be initialised "
                  "with load_stations_metadata object method.")
            return None

        # interpolation
        series = {}
        coords_to_interpolate = self.stations[['lat', 'lon']].itertuples()
        for station, inlat, inlon in coords_to_interpolate:
            points = self.nearest_points[station]
            p = [(lon, lat, values[i, j])
                 for i, j, lat, lon in zip(points["indices"]['ilat'],
                                           points["indices"]['jlon'],
                                           points["latitudes"],
                                           points["longitudes"])]
            series[station] = interpolation(inlon, inlat, p)
        return pd.Series(series)

    def _flon(self, i, j):
        return i*self.ccoord*self.c0+j*self.c1

    def _flat(self, i, j):
        return i*self.ccoord*self.c1+j*self.c0


def squared_dist(p1, p2):
    """Compute the squared distance between two points."""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def area_triangle(p0, p1, p2):
    """Compute the area between three points."""
    return 0.5*abs((p1[0]-p0[0])*(p2[1]-p0[1]) -
                   (p2[0]-p0[0])*(p1[1]-p0[1]))


def triangular_interpolation(x, y, points):
    """
    Interpolate (x,y) from values associated with four points.

    Compute an interpolation based on triangles for a (x, y) coordinate
    located inside a quadrilateral defined by four points.

    Parameters
    ----------
    x, y : float
        Coordinates of the input point.
    points : list of four float triplets
        Four points (x, y, value) forming a quadrilateral.
        First point must be the nearest neighbour.

    """
    (x0, y0, q0), (x1, y1, q1), (x2, y2, q2), (x3, y3, q3) = points

    areas = [area_triangle((x, y), points[0][0:2], points[1][0:2]),
             area_triangle((x, y), points[1][0:2], points[2][0:2]),
             area_triangle((x, y), points[2][0:2], points[3][0:2]),
             area_triangle((x, y), points[3][0:2], points[0][0:2])]

    v = (q0 * (areas[1]+areas[2]) + q1 * (areas[2]+areas[3]) +
         q2 * (areas[3]+areas[0]) + q3 * (areas[0]+areas[1]))
    v = v / (2. * sum(areas))

    return 0.5*(v + q0)


def bilinear_interpolation(x, y, points, sort=True):
    """
    Interpolate (x,y) from values associated with four points.

    Compute a bilinear interpolation for an (x, y) coordinate
    located inside a rectangle defined by four points.

    Parameters
    ----------
    x, y : float
        Coordinates of the input point.
    points : list of four float triplets
        Four points (x, y, value) forming a rectangle.
        The four points can be in any order.

    """
    # order points by x, then by y
    if sort is True:
        points = sorted(points)

    x, y = np.float32(x), np.float32(y)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    # verify args consistency
    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise evt.EvaltoolsError('points do not form a rectangle !!!')
    cond = (
        not round(np.float32(x1), 3) <= round(x, 3) <= round(np.float32(x2), 3)
        or not round(np.float32(y1), 3) <= round(y, 3) <=
        round(np.float32(y2), 3)
    )
    if cond:
        print(x, y, ' : ', x1, x2, y1, y2)
        for z in [x, y, x1, x2, y1, y2]:
            print(type(z))
        print('(x, y) not within the rectangle !!!')
        return np.nan
        # raise evt.EvaltoolsError('(x, y) not within the rectangle !!!')

    # interpolation
    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)


def interpolation(x, y, points):
    """Choose how to interpolate (x,y) according to grid type."""
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = \
        sorted(points)

    # regular grid
    if x1 == _x1 and x2 == _x2 and y1 == _y1 and y2 == _y2:
        return bilinear_interpolation(x, y, points)
    else:  # non-regular grid
        return triangular_interpolation(x, y, points)
