# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Define how evaltools is imported."""

import evaltools.utils
import evaltools.scores
import evaltools.timeseries
import evaltools.plotting
import evaltools.tables
import evaltools.quarter
import evaltools.interpolation
import evaltools.fairmode
import evaltools.netcdf
from evaltools.dataset import *
from evaltools.evaluator import *
import evaltools.sqlite

__version__ = '1.0.10'


class EvaltoolsError(Exception):
    """Error class for the package."""

    pass
