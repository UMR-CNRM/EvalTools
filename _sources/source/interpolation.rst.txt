Interpolation example
=====================


.. highlight:: python

Evaltools also has interpolation features. In
:mod:`interpolation<evaltools.interpolation>` module, you can find
tools to perform bilinear interpolation at lat/lon locations from grid values.
However, :mod:`interpolation<evaltools.interpolation>` can be easily replaced
by scipy or xarray.

First, we import a list of stations where we want interpolated values.

.. ipython:: python

    import numpy as np
    import evaltools as evt
    listing_path = "../doc/sample_data/listing"
    stations = evt.utils.read_listing(listing_path)

Let's define our domain boundaries and create arbitrarily simulated data.

.. ipython:: python

    min_lat = 45.
    min_lon = 6.
    d_lon = .5
    d_lat = .5
    nb_lon = 28
    nb_lat = 14

    sim_a = np.array(
        [
            [lat*lon for lon in range(nb_lon)]
            for lat in range(nb_lat)
        ]
    )
    sim_b = np.array(
        [
            [lat + lon for lon in range(nb_lon)]
            for lat in range(nb_lat)
        ]
    )


We now have a grid of values for two species A and B.

.. note:: Such grids could have been loaded from a netcdf or a grib file using
    libraries like `netCDF4` or `xarray`.


With xarray
-----------

.. ipython:: python

    import xarray as xr

    interp_lat = xr.DataArray(stations.lat, dims='station')
    interp_lon = xr.DataArray(stations.lon, dims='station')

    data = xr.DataArray(
        np.flipud(sim_a),
        dims=('lat', 'lon'),
        coords={
            'lat': np.arange(min_lat, min_lat + d_lat*nb_lat, d_lat),
            'lon': np.arange(min_lon, min_lon + d_lon*nb_lon, d_lon),
        },
    )

    data.interp(lat=interp_lat, lon=interp_lon)


With Scipy
----------

.. ipython:: python

    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (
            np.arange(min_lat, min_lat + d_lat*nb_lat, d_lat),
            np.arange(min_lon, min_lon + d_lon*nb_lon, d_lon),
        ),
        np.flipud(sim_a),
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    interpolator(
        stations[['lat', 'lon']].values
    )


With evaltools
--------------

We need to create a :class:`Grid<evaltools.interpolation.Grid>` object,
load metadata from a station list file and load one list of stations per
species where interpolation will be performed (in our case we will take the
first five stations of the `stations` variable defined above for species A
and the last five for species B).

.. ipython:: python

    interpolator = evt.interpolation.Grid(
        min_lat, min_lon, d_lon, d_lat, nb_lon, nb_lat,
    )
    interpolator.load_stations_metadata(listing_path)
    interpolator.load_station_lists(
        {'A': stations.index[:5], 'B': stations.index[-5:]}
    )

All that remains to be done is to load the input arrays and perform
the interpolation.

.. ipython:: python

    interpolator.set_grids({'A': sim_a, 'B': sim_b})
    interpolator.interpolate()

.. note:: Before performing the interpolation, you could
    use :meth:`view<evaltools.interpolation.Grid.view>` method to
    visualize input array and check that values are not upside down.

.. note:: If you then want to perform interpolation from other input arrays
    for the same species you don't have to load station lists again but only
    updating the arrays by calling
    :meth:`set_grids<evaltools.interpolation.Grid.set_grids>` method again.
