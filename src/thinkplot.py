# coding: utf-8

"""
This file contains code for use with "Think Stats", by Allen B.Downey, available from
greenteapress.com
"""

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

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
        warnings.warn('Ran out of colors. Starting over.')
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
        options['align'] = 'edge'
    elif options['align'] == 'right':
        options['align'] = 'edge'
        options['width'] *= -1

    Bar(xs, ys, **options)


LEGEND = True
LOC = None


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


def Clf():
    """
    Clears the figure and any hists that have been set.
    """
    global LOC
    LOC = None
    _Brewer.ClearIter()
    pyplot.clf()
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)


def Show(**options):
    """
    Shows the plot.
    For options, see Config.
    :param options: keyword args used to inovke various pyplot functions
    """
    clf = options.pop('clf', True)
    Config(**options)
    pyplot.show()
    if clf:
        Clf()


def Plotly(**options):
    """
    Shows the plot
    For options, see Config
    :param options: keyword args used to invoke various pyplot functions
    """
    clf = options.pop('clf', True)
    Config(**options)
    import plotly.plotly as plotly
    url = plotly.plot_mpl(pyplot.gcf())
    if clf:
        Clf()
    return url


def Save(root=None, formats=None, **options):
    """
    Save the plot in the given formats and clears the figure.
    For options, see Config.
    :param root: string filename root
    :param formats: list of string formats
    :param options: keyword args used to invoke various pyplot functions
    """
    clf = options.pop('clf', True)
    Config(**options)

    if formats is None:
        formats = ['pdf', 'eps']

    try:
        formats.remove('plotly')
        Plotly(clf=False)
    except ValueError:
        pass

    if root:
        for fmt in formats:
            SaveFormat(root, fmt)

    if clf:
        Clf()


def SaveFormat(root, fmt='eps'):
    """
    Writes the current figure to file in the given format.
    :param root: string filename root
    :param fmt: string format
    """
    filename = '%s.%s' % (root, fmt)
    print('writing', filename)
    pyplot.savefig(filename, format=fmt, dpi=300)


def PrePlot(num=None, rows=None, cols=None):
    """
    Takes hints about what's coming.
    :param num: number of lines that will be plotted
    :param rows: number of rows of subplots
    :param cols: number of columns of subplots
    """
    if num:
        _Brewer.InitIter(num)

    if rows is None and cols is None:
        return

    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    # resize the image, depending on the number of rows and cols
    size_map = {(1, 1): (8, 6),
                (1, 2): (12, 6),
                (1, 3): (12, 6),
                (2, 2): (10, 10),
                (2, 3): (16, 10),
                (3, 1): (8, 10),
                (4, 1): (8, 12)}

    if (rows, cols) in size_map:
        fig = pyplot.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    # create the first subplot
    if rows > 1 or cols > 1:
        ax = pyplot.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols
    else:
        ax = pyplot.gca()

    return ax


def Pmf(pmf, **options):
    """
    Plots a Pmf or Hist as a line.
    :param pmf: Hist or Pmf object
    :param options: keyword args passed to pyplot.plot
    """
    xs, ys = pmf.Render()
    # low, high = min(xs), max(xs)

    width = options.pop('width', None)
    if width is None:
        try:
            width = np.diff(xs).min()
        except TypeError:
            warnings.warn("Pmf: Can't compute bar width automatically."
                          "Check for non-numeric types in Pmf."
                          "Or try providing width option.")
    points = []
    lastx = np.nan
    lasty = 0
    for x, y in zip(xs, ys):
        if (x - lastx) > 1e-5:
            points.append((lastx, 0))
            points.append((x, 0))
        points.append((x, lasty))
        points.append((x, y))
        points.append((x+width, y))

        lastx = x + width
        lasty = y
    points.append((lastx, 0))

    pxs, pys = zip(*points)
    print("**" * 10)
    print(xs)
    print(ys)
    print(pxs)
    print(pys)
    print("**" * 10)

    align = options.pop('align', 'center')
    if align == 'center':
        pxs = np.array(pxs) - width /2.0
    if align == 'right':
        pxs = np.array(pxs) - width

    options = _Underride(options, label=pmf.label)
    Plot(pxs, pys, **options)


def Plot(obj, ys=None, style='', **options):
    """
    Plots a line.
    :param obj: sequence of x values, or Series, or anything wiht Render()
    :param ys: sequence of y values
    :param style: style string passed along to pyplot.plot
    :param options: keyword args passed to pypolt.plot
    """
    options = _UnderrideColor(options)
    label = getattr(obj, 'label', '_nolegend_')
    options = _Underride(options, linewidth=1.5, alpha=0.7, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, 'Render'):
            xs, ys = obj.Render()
        if isinstance(obj, pd.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        pyplot.plot(xs, style, **options)
    else:
        pyplot.plot(xs, ys, style, **options)


def Pmfs(pmfs, **options):
    """
    Plots a sequence of PMFs.
    Options are passed along for all PMFs. If you want different options for each pmf,
    make multiple calls to Pmf.
    :param pmfs: sequence of PMF objecs
    :param options: keyword args passed to pyplot.plot
    """
    for pmf in pmfs:
        Pmf(pmf, **options)


def SubPlot(plot_number, rows=None, cols=None, **options):
    """
    Configures the number of subplots and changes the current plot.
    :param plot_number: int
    :param rows: int
    :param cols: int
    :param options: passed to subplot
    """
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    return pyplot.subplot(rows, cols, plot_number, **options)


