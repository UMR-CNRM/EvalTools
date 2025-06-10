# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Module gathering the functions used to sort scores values."""


def identity(x):
    """
    Return the identical value.

    Used to sort score values where the higher is the better.

    """
    return x


def negate(x):
    """
    Return the opposite value.

    Used to sort score values where the lower is the better.

    """
    return -x


def abs_to_one(x):
    """
    Return the absolute of x-1.

    Used to sort score values where the closer to one is the
    better.

    """
    return abs(x - 1)


def absolute(x):
    """
    Return the absolute value.

    Used to sort score values where the higher in absolute terms is the
    better.

    """
    return abs(x)
