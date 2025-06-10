# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
_scoreobj : module that implements the class `Score`.

It is designed to represent the scores and
statistic functions used by evaltools.

To be valid, a score compare observations
and simulation, have a sorting key, and
have units.

The class Score gives a framework to
access those attributes and method
in a convienient way.

"""
from . import _checks
from ._checks import ScoreImplementationError
from functools import wraps

# This dictionnary will be filled at import
IMPLEMENTED_SCORES = {}


class Score(object):
    """
    Class representing the scores.

    Attributes
    ----------
    units : str
        Units of the score. None if dimensionless and
        "{field_units}" if the score has the units of the data.
    name : str
        Name of the score.
    score_func : callable
        Callable that computes the score.
    sort_key : callable
        Callable to sort the score values from best to worst in performance.

    Methods
    -------
    __call__ : calls `score_func`


    """

    def __init__(self, func, name, units=None, sort_key=None):
        """Build a Score object."""
        self.units = units
        self.name = name
        self.score_func = _checks.validate_score_func(func)
        self._have_field_units = "{field_units}" in self.units
        self._have_extra_params = len(_checks.signature(func).parameters) > 2

        if sort_key is not None:
            self.sort_key = sort_key
        else:
            self.sort_key = lambda x: x

    @property
    def units(self):
        """Get score units."""
        if self._units is not None:
            return self._units
        else:
            return '[dim. less]'

    @units.setter
    def units(self, value):
        """Set score units."""
        self._units = value

    @property
    def name(self):
        """Get score name."""
        if self._name is not None:
            return self._name
        else:
            return 'Unknown score name'

    @name.setter
    def name(self, value):
        """Set score name."""
        self._name = value

    def __repr__(self):
        """Get score string representation."""
        return f"Score(name={self.name}, units={self.units})"

    def __str__(self):
        """Get score string formatting."""
        return f"{self.name} {self.units}"

    def __call__(self, obs, sim, **kw):
        """Call score function."""
        return self.score_func(obs, sim, **kw)


def score(name=None, units=None, alias=None, sort_key=None):
    """
    Decorate a function to check if it complies to the rules of valid score.

    Parameters
    ----------
    name : str
        Need to be a valid unused names (case-unsensitive).
    units : str
        Units of the score. None if the score is dimensionless and
        {field_units} if the score has the units of the data.
    alias : str
        Alias for the score.

    Raises
    ------
    ScoreImplementationError

    """
    def decorator(func):
        score_func = Score(func, name, units, sort_key)
        if name.lower() in IMPLEMENTED_SCORES.keys():
            raise ScoreImplementationError(
                f"The name {name} is already used. Please choose another one."
            )

        IMPLEMENTED_SCORES[name.lower()] = score_func
        if alias is not None:
            if alias.lower() in IMPLEMENTED_SCORES.keys():
                raise ScoreImplementationError(
                    f"The alias {alias} is already used." +
                    "Please choose another one."
                )
            IMPLEMENTED_SCORES[alias.lower()] = score_func

        if score_func._have_extra_params:
            _checks.params_annotations(score_func.score_func)

        return wraps(func)(score_func)

    return decorator
