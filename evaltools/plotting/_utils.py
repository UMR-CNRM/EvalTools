# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Tools to manage plotting functions."""
from functools import wraps
import inspect
from ._mpl import set_axis_elements, save_figure

IMPLEMENTED_PLOTS = dict()


def remove_prefix(text, prefix):
    """
    Remove the prefix of a string.

    Defined here for python < 3.9.

    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def add_kwargs_to_sig(**kwargs):
    """Decorate a function to add keyword arguments to its signature."""
    def decorator(func):
        for kw in kwargs:
            oldsig = inspect.signature(func)
            # search if a VAR_POSITIONAL or VAR_KEYWORD is present
            # if yes insert new parameter before it,
            # else insert it in last position
            params = list(oldsig.parameters.values())
            for i, param in enumerate(params):
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    break
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    break
            else:
                i = len(params)
            # add '_' to new parameter's name if already present
            name = kw
            while name in oldsig.parameters:
                name += '_'
            newparam = inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=kwargs[kw],
            )
            params.insert(i, newparam)
            # we can now build the signature for the wrapper function
            sig = oldsig.replace(parameters=params)
            func.__signature__ = sig

        return func

    return decorator


def common_doc_str(func):
    """Decorate plotting functions to add them a common docstr."""
    common_docstr = """
    annotation : str or None
        Additional information to write in the upper left corner of the plot.
    output_file : str or None
        File where to save the plots (without extension). If None, the figure
        is shown in a popping window.
    file_formats : list of str
        List of file extensions.
    fig : None or matplotlib.figure.Figure
        Figure to use for the plot. If None, a new figure is created.
    ax : None or matplotlib.axes._axes.Axes
        Axis to use for the plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object of the produced plot. Note that if the plot has been
        shown in the user interface window, the figure and the axis will not
        be usable again.
    matplotlib.axes._axes.Axes
        Axes object of the produced plot.

    """
    func.__doc__ = f"{func.__doc__.rstrip()}{common_docstr}"
    return func


def plot_func(func):
    """Decorate plotting functions to set up common parameters."""
    @add_kwargs_to_sig(output_file=None, file_formats=['png'])
    @add_kwargs_to_sig(annotation=None)
    @common_doc_str
    @wraps(func)
    def wrapper(
            *args,
            output_file=None, file_formats=['png'], annotation=None,
            **kwargs):

        res = func(*args, **kwargs)

        if annotation:
            set_axis_elements(
                res[1],
                annotate_kw=(
                    dict(
                        text=annotation, xy=(0, 1), xycoords='figure fraction',
                        va='top', fontsize='small', fontstyle='italic',
                        color='#7A7A7A',
                    )
                    if annotation
                    else None
                )
            )

        # set_axis_elements(title=title)

        save_figure(
            output_file,
            file_formats,
            bbox_inches=None,
            tight_layout=False,
        )
        return res

    IMPLEMENTED_PLOTS[remove_prefix(func.__name__, 'plot_')] = wrapper

    return wrapper
