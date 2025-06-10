# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Check arguments of score functions."""
from inspect import signature, _empty

DEV_MODE = False

if DEV_MODE:
    print('Running in dev mode')


class ScoreImplementationError(Exception):
    """Custom error for this module."""

    pass


def params_annotations(f):
    """Check score function annotations."""
    if not DEV_MODE:
        return

    sig = signature(f)
    if any([
        p.annotation is _empty for n, p in sig.parameters.items()
        if n not in ["obs", "sim"]
    ]):
        raise ScoreImplementationError(
            f"Missing annotations for parametered score {f.__name__}"
        )

    if any([
        not isinstance(p.annotation, type) for n, p in sig.parameters.items()
        if n not in ["obs", "sim"]
    ]):
        raise ScoreImplementationError(
            f"Some annotations are not valid types in {f.__name__}"
        )


def validate_score_func(f):
    """Check score function arguments."""
    if not f.__code__.co_varnames[:2] == ('obs', 'sim'):
        raise ScoreImplementationError(
            f"{f.__name__} first arguments must be 'obs' and 'sim'"

        )

    return f
