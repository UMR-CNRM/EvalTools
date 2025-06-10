.. evaltools documentation master file, created by
   sphinx-quickstart on Thu Jan 11 11:46:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to evaltools' documentation!
=====================================

**Version:** |version|

The Python package `evaltools` is designed to assess surface atmosphere
composition prediction models regarding to in-situ observations. This package
provides different tools to compute model scores and plot them. It is used for
evaluation of air quality models of Copernicus Atmosphere Monitoring Service
(CAMS).

The concept of `evaltools` is to compare observations (measured over time in fixed lat/lon
locations) to simulations (that can have a forecast horizon of several days) computed over
a period of several days. Therefore, it can be suited for other data types like AERONET
data, but will not manage data with a vertical component.

This software is governed by the CeCILL-C license under French law and abiding
by the rules of distribution of free software. You can use, modify and/ or
redistribute the software under the terms of the CeCILL-C license as
circulated by CEA, CNRS and INRIA at the following URL “http://www.cecill.info”.

.. toctree::
    :maxdepth: 3

    Installation <source/installation>
    User guide <source/guide>
    API reference <source/evaltools>

:ref:`genindex`
