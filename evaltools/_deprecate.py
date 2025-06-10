# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Define decorators used to deprecate functions or arguments."""
import warnings
from functools import wraps
from functools import partial


def deprecate(
        name,
        alternative,
        stacklevel=2,
        msg=None):
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    stacklevel : int, default 2
        Argument passed to warnings.warn.
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'

    """
    alt_name = alternative.__name__
    wn = FutureWarning
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    @wraps(alternative)
    def wrapper(*args, **kwargs):
        warnings.warn(warning_msg, wn, stacklevel=stacklevel)
        return alternative(*args, **kwargs)

    return wrapper


def deprecate_func(
        name,
        alternative,
        stacklevel=2,
        msg=None):
    """
    Decorate a function to emit a deprecation warning on use.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    stacklevel : int, default 2
        Argument passed to warnings.warn.
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'

    """
    alt_name = alternative.__name__
    wn = FutureWarning
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    def _deprecate_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(warning_msg, wn, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return _deprecate_func


def deprecate_kwarg(
        old_arg_name,
        new_arg_name,
        stacklevel=2):
    """
    Decorate a function to deprecate a keyword argument.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    stacklevel : int, default 2
        Argument passed to warnings.warn.

    """
    def _deprecate_kwarg(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {repr(old_arg_name)}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated, "
                        f"use {repr(new_arg_name)} instead."
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {repr(old_arg_name)} "
                        f"or {repr(new_arg_name)}, not both."
                    )
                    raise TypeError(msg)
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return wrapper

    return _deprecate_kwarg


def deprecate_attrs(**kwargs):
    """
    Add deprecated attributes to a class.

    The decorator factory returns a class decorator that, for each
    deprecated attribute, adds a property accessing the new attribute
    and printing a future warning.

    Parameters
    ----------
    kwargs: dict
        Keyword arguments where the keys are the recommended attributes and
        the values are the deprecated names.

    """
    def _add_properties(cls):
        def func(self, new, old):
            warning_msg = f"{old} is deprecated, use {new} instead."
            warnings.warn(warning_msg, FutureWarning, stacklevel=2)
            return getattr(self, new)

        for new, old in kwargs.items():
            prop = property(partial(func, new=new, old=old))
            prop.__doc__ = "Deprecated."
            setattr(cls, old, prop)

        return cls

    return _add_properties
