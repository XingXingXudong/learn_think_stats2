# coding: utf-8

"""
This file contain code for use with "Think Stats" and "Think Bayes", both by Allen B. Downey,
available form greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import re
import pandas as pd
import numpy as np
import logging
from collections import Counter
import copy
import math


class FixeWidthVariables(object):
    """
    Represents a set of variables in a fixed width file.
    """
    def __init__(self, variables, index_base=0):
        """
        Initializes.
        :param variables: DataFrame
        :param index_base: are the indices 0 or 1 based?
        Attributes:
        colspecs: lsit of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables
        # note: by default, subtract 1 from colspecs
        self.colspecs = variables[['start', 'end']] - index_base
        # convert colspecs to a list of pair of int
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = variables['name']

    def read_fixd_width(self, filename, **options):
        """
        Read a fixed width ASCII file.
        :param filename: string filename
        :param options: kv options for open()
        :return: DataFrame
        """
        df = pd.read_fwf(filename, colspecs=self.colspecs, names=self.names, **options)
        return df


def read_stata_dct(dct_file, **option):
    """
    Read a Stata dictionary file.
    :param dct_file: doc
    :param option:
    :return:
    """
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)

    var_info = []

    for line in open(dct_file, **option):
        # match = re.search(r'_column\(([^)]*)\)', line)
        # match = re.search(r'_column\(([^)]*)\)', line)
        match = re.search(r'_column\(([\d]*)\)', line)
        if match:
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pd.DataFrame(var_info, columns=columns)

    # fill in the end column by shifting the start column
    variables['end'] = variables.start.shift(-1)
    variables.loc[len(variables)-1, 'end'] = 0
    dct = FixeWidthVariables(variables, index_base=1)
    return dct


class _DictWrapper(object):
    """
    An object that contains a dictionary.
    """
    def __init__(self, obj=None, label=None):
        """
        Initializes the distribution.
        :param obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs
        :param label: string label
        """
        self.label = label if label is not None else '_nolegend_'
        self.d = {}

        # flag whether the distribution is under a log transform
        self.log = False

        if obj is None:
            return

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.labesl = label if label is not None else obj.label

        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.d.update(obj.Items())
        elif isinstance(obj, pd.Series):
            self.d.update(obj.values_counts().iteritems())
        else:
            # finally, treat it like a list
            self.d.update(Counter(obj))

        # if len(self) > 0 and isinstance(self, Pmf):
        #     self.Normalize()

    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        return '%s(%s)' % (cls, str(self.d))

    __repr__ = __str__

    def __eq__(self, other):
        return self.d == other.d

    def __iter__(self):
        return iter(self.d)

    def __iterkeys__(self):
        """
        Returns an iterator over keys
        :return: iterator
        """
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, value):
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def Copy(self, label=None):
        """
        Return a copy
        Make a shallow copy of d. If you want a deep copy of d,
        use copy.deepcopy on teh whole object.
        :paramlabel: string label for teh new Hist
        :returns: new _DictWrapper with the same type
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new

    def scale(self, factor):
        """
        Multiplies the value by a factor.
        :param factor: what to multiply by
        :return: new object
        """
        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        """
        Log transforms the probabilities.
        Removes values with probability 0.
        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        """
        Exponentiates the probabilities.
        :param m: how much to thift the ps before exponentiating
        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        """Gets teh dictionary."""
        return self.d

    def SetDict(self, d):
        """Sets the dictionary."""
        self.d = d

    def Values(self):
        """
        Gets an unsorted sequence of values.
        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def Items(self):
        """Gets an unstorted sequence of (value, freq/prob) pairs"""
        return self.d.items()

    def Render(self, **options):
        """
        Generates a sequence of points suitable for plotting.
        Note: options are ignored.
        :param options: ignored
        :return: tuple of (sorted value sequence, freq/prob sequence
        """
        if min(self.d.keys()) is np.nan:
            logging.warning('Hist: contains NaN, may not render correctly')

        return zip(*sorted(self.Items()))

    def MakeCdf(self, label=None):
        """Make a Cdf."""
        label = label if label is None else self.label
        return Cdf(self, label=label)

    def Print(self):
        """Prints the values and freq/prob in ascending order"""
        for val, prob in sorted(self.d.items()):
            print(val, prob)

    def Set(self, x, y=0):
        """
        Sets the freq/prob associated with the value x.
        :param x: number value
        :param y: number freq or prob
        """
        self.d[x] = y

    def Incr(self, x, term=1):
        """
        Increments the freq/prob associated with the value x.
        :param x: number value
        :param term: how much to incrment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        """
        Scales the freq/prob associated with the values x.
        :param x: number value
        :param factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        """
        Removes a value
        Throws an exception if the value is not there
        :param x: value to remove
        """
        del self.d[x]

    def Total(self):
        """Returns the total of the frequencies/probabilities in the map"""
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        """Return the largest frequency/probability in the map."""
        return max(self.d.values())

    def Largest(self, n=10):
        """
        Returns the largest n values, with frequency/probability.
        :param n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]

    def Smallest(self, n=10):
        """
        Returns the smallest n values, with frequency/probability.
        :param n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]


class Hist(_DictWrapper):
    """
    Represents a histogram, which is a map from values to frequencies.
    Values can be any hashable type; frequencies are integer counters.
    """
    def Freq(self, x):
        """
        Gets the ferquency associated with the value x.
        :param x: number value
        :return: int frequency
        """
        return self.d.get(x, 0)

    def Freqs(self, xs):
        """Get frequencies for a sequence of values."""
        return [self.Freq(x) for x in xs]

    def IsSubset(self, other):
        """Checks wheter the values in this histogram are a subset of the values in the given histogram"""
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        """Subtracts the values in the given histogram from this histogram"""
        for val, freq in other.Items():
            self.Incr(val, -freq)


class Pdf(_DictWrapper):
    pass


class Cdf(_DictWrapper):
    pass


class Pmf(_DictWrapper):
    def Normalize(self):
        pass


class Test(object):
    pass

class Test1(object):
    pass


if __name__ == "__main__":
    hist = Hist([1, 2, 2, 3, 5])
    x, y = hist.Render()
    print(x)
    print(y)

