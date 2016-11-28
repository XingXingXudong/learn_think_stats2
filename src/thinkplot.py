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
    """
    Encapsulates a nice sequence of colors

    Shades of blue that look lood in color and canbe distinguishe in grayscale(up to point).
    Borrowed fro m http://colorbrewer2.org
    """
    color_iter = None
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecael', '#4292c6',
              '#2171b5', '#08519c', '#08306b'][::-1]
    # lists that indicate which colors to use depending on how many are used
    which_colors = [[],
                    [1],
                    [1, 3],
                    [0, 2, 4],
                    [0, 2, 4, 6],
                    [0, 2, 3, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    current_figure = None

    @classmethod
    def Colors(cls):
        """
        Returns the lsit of colors.
        :return: colors
        """
        return cls.colors

    @classmethod
    def ColorGenerator(cls, num):
        """
        Returns an iterator of color strings.
        :param num: how many colors will be used
        :return: colors
        """
        for i in cls.which_colors[num]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in _Brewer')

    @classmethod
    def InitIter(cls, num):
        """
        Initializes the color iterator with the given number of colors.
        :param num: how many colors will be used
        """
        cls.color_iter = cls.ColorGenerator(num)

    @classmethod
    def ClearIter(cls):
        """Sets the color iterator to None."""
        cls.color_iter = None

    @classmethod
    def GetIter(cls, num):
        """Gets the color iterator."""
        fig = pyplot.gcf()  # pyplot.gcf()get a reference to the current figure.
        if fig != cls.current_figure:
            cls.InitIter(num)
            cls.current_figure = fig

        if cls.color_iter is None:
            cls.InitIter(num)

        return cls.color_iter


def Bar(xs, ys, **options):
    """
    Plots a line.
    :param xs: sequence of x value
    :param ys: sequence of y value
    :param options: keyword args passed to pyplot.bar
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.6)
    pyplot.bar(xs, ys, **options)


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


def Config(**options):
    """
    Configures the plot.
    Pull options out of the options dictionary and passes them to the corresponding pyplot functions.
    :param options: keyword args to pyplot functions
    """
    names = ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
             'xticks', 'yticks', 'axis', 'xlim', 'ylim']

    for name in names:
        if name in options:
            getattr(pyplot, name)(options[name])

    global LEGEND
    LEGEND = options.get('legend', LEGEND)

    if LEGEND:
        global LOC
        LOC = options.get('loc', LOC)
        pyplot.legend(loc=LOC)

    val = options.get('xticklabels', None)


def Show(**options):
    """
    Shows the plot.
    For options, see Config.
    :param options: keyword args used to inovke various pyplot functions
    """
    clf = options.pop('clf', True)
    Config(**options)
    pylot.show()
    if clf:
        Clf()





