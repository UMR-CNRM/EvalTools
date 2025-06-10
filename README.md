# *evaltools* : a model evaluation Python package

The Python package *evaltools* is designed to assess surface atmosphere
composition prediction models with respect to in-situ observations. This
package provides different tools to compute and plot model scores. It is used
to evaluate air quality models from the Copernicus Atmosphere Monitoring
Service (CAMS).

The concept of *evaltools* is to compare observations (measured over time
at fixed lat/lon locations) with simulations (which may have a forecast
horizon of several days) computed over a period of several days. It can
therefore be used for other data types like AERONET data, but will not
handle data with a vertical component.

## Installation

Several python libraries are required to run evaltools. To install them,
you can do
```bash
pip3 install -r requirements.txt
```

Optionally, you can also install
```bash
pip3 install -r optional-requirements.txt
```
which are dependencies needed for plotting maps or build the documentation.

The version numbers specified inside these two requirement files are the one
that have been used to test the library.

Then, go to the package directory and execute

```bash
pip3 install . --no-build-isolation
```

## Documentation

Online doc of the latest release on master branch is available at https://umr-cnrm.github.io/EvalTools

To compile html documentation, draw example charts with

```bash
make test
```
and then compile the html documentation with

```bash
make doc
```

To look at the documentation then open `doc/_build/html/index.html` with
a web navigator.

## License

Copyright Météo France (2017 - )

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software. You can use, modify
and/ or redistribute the software under the terms of the CeCILL-C license as
circulated by CEA, CNRS and INRIA at the following
URL “http://www.cecill.info”.
