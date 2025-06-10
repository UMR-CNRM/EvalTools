"""
Example of use of evaltools.interpolation.

However, evaltools.interpolation can be easily replaced by scipy:

```
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
    vals = interpolator(
        stations[['lat', 'lon']].values
    )
```

or xarray:
```
    import xarray as xr

    lat = np.arange(min_lat, min_lat + d_lat*nb_lat, d_lat)
    lon = np.arange(min_lon, min_lon + d_lon*nb_lon, d_lon)
    da = xr.DataArray(
        np.flipud(sim_a),
        coords={'lat': lat, 'lon': lon},
    )
    obs_lat = xr.DataArray(stations.lat, dims='obs')
    obs_lon = xr.DataArray(stations.lon, dims='obs')
    da.interp(lat=obs_lat, lon=obs_lon)
```

"""
import numpy as np
import evaltools as evt

# listing
listing_path = "../doc/sample_data/listing"
stations = evt.utils.read_listing(listing_path)

# domain definition
min_lat = 45.
min_lon = 6.
d_lon = .5
d_lat = .5
nb_lon = 28
nb_lat = 14

# simulate data
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

# Grid object
interpolator = evt.interpolation.Grid(
    min_lat, min_lon, d_lon, d_lat, nb_lon, nb_lat,
)
interpolator.load_stations_metadata(listing_path)
interpolator.load_station_lists(
    {'A': stations.index[:5], 'B': stations.index[-5:]}
)
interpolator.set_grids({'A': sim_a, 'B': sim_b})

# visualisation
interpolator.view()

# interpolation
interpolator.interpolate()
