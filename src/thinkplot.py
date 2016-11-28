# coding: utf-8

"""
This file contains code for use with "Think Stats", by Allen B.Downey, available from
greenteapress.com
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot

import warnings


def _Underride(d, **options):
    """
    Add key-value pairs to d only if key is not in d.
    If d is None, create a new dictionary.
    :param d: dictionary
    :param options: keyword args to add to d
    :return:
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def _UnderrideColor(options):
    """
    If color is not in the options, choose a color.
    :param options: keyword args passed to pyplot.bar
    :return options
    """
    if 'color' in options:
        return options

    # get the current color iterator; if there is none, init one
    color_iter = _Brewer.GetIter(5)

    try:
        options['color'] = next(color_iter)
    except StopIteration:
        # if you run out of colors, initialize the color iterator and try again.
        warnings.warn('Run out of colors. Starting over.')
        _Brewer.ClearIter()
        _UnderrideColor()

    return options


class _Brewer(object):
    pass



def Bar(xs, ys, **options):
    """
    Plots a line.
    :param xs: sequence of x value
    :param ys: sequence of y value
    :param options: keyword args passed to pyplot.bar
    """
    options = _UnderrideColor(options)



def Hist(hist, **options):
    """
    Plot a Pmf or Hist with a bar plot.
    The default width of the bars is based on the minimum difference between values in the
    Hist. If that's too small, you can override it by provaiding a width keyword argument,
    in the same units
    :param hist: Hist or Pmf object
    :param options: keyword args passed to pyplot.bar
    """
    # find the minimum distance between adjacent values
    xs, ys = hist.Render()

    if 'width' not in options:
        try:
            options['width'] = 0.9 * np.diff(xs).min()
        except TypeError:
            warnings.warn("Hist: Can't compute bar width automatically."
                          "Check for non-numeric types in Hist."
                          "Or try providing width option")

    options = _Underride(options, label=hist.label)
    options = _Underride(options, align='center')
    if options['align'] == 'left':
        options['align'] == 'edge'
    elif options['align'] == 'right':
        options['align'] == 'edge'
        options['width'] *= -1

    Bar(xs, ys, **options)


