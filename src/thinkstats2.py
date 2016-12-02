# coding: utf-8

"""
This file contain code for use with "Think Stats" and "Think Bayes", both by Allen B. Downey,
available form greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import bisect
import re
import pandas as pd
import numpy as np
import logging
from collections import Counter
import copy
import math
import random


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
            self.d.update(obj.value_counts().iteritems())
        else:
            # finally, treat it like a list
            self.d.update(Counter(obj))

        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()


    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        return '%s(%s)' % (cls, str(self.d))

    __repr__ = __str__

    def __eq__(self, other):
        return self.d == other.d

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
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

    def Render(self):
        """
        Generates a sequence of points suitable for plotting.
        Note: options are ignored.
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
        return sorted(self.d.items(), reverse=False)[:n]


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


def CohenEffectSize(group1, group2):
    """
    Compute Cohen's d.
    :param group1: Series or NumPy array
    :param group2: Series or NumPy array
    :return: float, the Cohen'd
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)  # 这里计算的方式与wikipedia上的稍有不同
    d = diff/math.sqrt(pooled_var)
    return d


class Pdf(_DictWrapper):
    pass


class Cdf(object):
    """
    Represents a cumulative distribution function.
    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        label: string used as a graph label.
    """
    def __init__(self, obj=None, ps=None, label=None):
        """
        Initializes.
        If ps is provided, obj must be the corresponding list of values.
        :param obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs
        :param ps: list of cumulative probabilities.
        :param label: string label
        """
        self.label = label if label is not None else '_nolegend_'

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            if not label:
                self.label = label if label is not None else obj.label

        if obj is None:
            # caller does not provides obj, make an empty Cdf.
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            if ps is not None:
                logging.warning("Cdf: can't pass ps without also passing xs.")
            return
        else:
            # if the caller provide xs and ps, just store them
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Cdf: ps can't be a string")

                    self.xs = np.asarray(obj)
                    self.ps = np.asarray(ps)
                    return

        # caller has provided just obj, not ps
        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)
            return

        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)

        if len(dw) == 0:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            return

        xs, freqs = zip(*sorted(dw.Items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=np.float)
        self.ps /= self.ps[-1]

    def __str__(self):
        return 'Cdf(%s, %s)' % (str(self.xs), str(self.ps))

    __repr__ = __str__

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, x):
        return self.Prob(x)

    def __setitem__(self):
        raise UnimplementedMethodException()

    def __delitem__(self):
        raise UnimplementedMethodException()

    def __eq__(self, other):
        return np.all(self.xs == other.xs) and np.all(self.ps == other.ps)

    def Copy(self, label=None):
        """
        Returns a copy of this Cdf
        :param label: string label for thre new Cdf
        """
        if label is None:
            label = self.label
        return Cdf(list(self.xs), list(self.ps), label=label)

    def MakePmf(self, label=None):
        """Makes a Pmf."""
        if label is None:
            label = self.label
        return Pmf(self, label=label)

    def Values(self):
        """
        Returns a sorted list of values
        """
        return self.xs

    def Items(self):
        """
        Returns a sorted sequence of (value, probability) pairs.
        Note: in Python3, returns an iterator.
        """
        a = self.ps
        b = np.roll(a, 1)
        b[0] = 0
        return zip(self.xs, a-b)

    def Shift(self, term):
        """
        Adds a term to the xs.
        :param term: how much to add
        """
        new = self.Copy()
        # don't use +=, or else an int array + float yields int array
        new.xs = new.xs + term
        return new

    def Scale(self, factor):
        """
        Multiplies the xs ba a factor.
        :param factor: what to multiply by
        """
        new = self.Copy()
        # don't use *=, or else an int array * float yields int array
        new.xs = new.xs * factor
        return new

    def Prob(self, x):
        """
        Returns CDF(x), the probability that corresponds to value x.
        :param x: number
        :return: float probability
        """
        if x < self.xs[0]:
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def Probs(self, xs):
        """
        Gets probabilities for a sequence of values.
        :param xs: any sequence that can be converted to Numpy array
        :return: Numpy array of cumulative probabilities.
        """
        xs = np.array(xs)
        index = np.searchsorted(self.xs, xs, side='right')
        ps = self.ps[index-1]
        ps[xs < self.xs[0]] = 0.0
        return ps

    ProbArray = Probs

    def Value(self, p):
        """
        Returns InverseCDF(p), the value that corresponds to probability p.
        :param p: number in the range [0,1]
        :return: number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        index = bisect.bisect_left(self.ps, p)
        return self.xs[index]

    def ValueArray(self, ps):
        """
        Returns InverseCDF(p), the value that corresponds to probability p.
        :param ps: NumPy array of numbers
        :return: Numpy array of values
        """
        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('Probability p must be in range [0, 1]')

        index = np.searchsorted(self.ps, ps, side='left')
        return self.xs[index]

    def Percentile(self, p):
        """
        Returns the value that corresponds to percentile p
        :param p: number in the range [0, 100]
        :return: number value
        """
        return self.Value(p / 100.0)

    def PercentileRank(self, x):
        """
        Returns the percentile rank of the value x.
        :param x: potential value in the CDF
        :return: percentile rank in the range 0 to 100
        """
        return self.Prob(x) * 100.0

    def Random(self):
        """Choses a random value from this distribution."""
        return self.Value(random.random())

    def Sample(self, n):
        """
        Generates a random sample from this distribution
        :param n: int length of the sampel
        :return: Numpy array
        """
        ps = np.random.random(n)
        return self.ValueArray(ps)

    def Mean(self):
        """
        Computes the mean of a CDF
        :return: float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def CredibleInterval(self, percentage=90):
        """
        Computes the central credible interval
        If percentage=90, computes the 90% CI.
        :param percentage: float between 0 and 100
        :return: sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0)
        interval = self.Value(prob), self.Value(1-prob)
        return interval

    ConfidenceInterval = CredibleInterval

    def _Round(self, multiplier=1000.):
        """
        An entry is added to the cdf only if the percentile differs from the previous
        value in a significant digit, where the number of significant digits is determined
        by multiplier. The default is 1000, which kepes log10(1000) = 3 significant digits.
        :param multiplier:
        :return:
        """
        # todo (write this method)
        raise UnimplementedMethodException

    def Render(self, **options):
        """
        Generates a sequence of points suitable for plotting.
        An empirical CDF is a step function; linear interpolation can be misleading
        :param options: options are ignored
        """
        def interleave(a, b):
            c = np.empty(a.shape[0] + b.shape[0])
            c[::2] = a
            c[1::2] = b
            return c
        


class Pmf(_DictWrapper):
    """
    Represents a probability mass function.
    Valuse can be any hashable type; probabilities are floating-point.
    Pmf are not necessarily normalized.
    """
    def Prob(self, x, default=0):
        """
        Gets the probability associated with the value x.
        :param x: number value
        :param default: value to return if the key is not there
        :return: flota probablity
        """
        return self.d.get(x, default)

    def Probs(self, xs):
        """Gets probabilities for a sequence of values."""
        return [self.Probs(x) for x in xs]

    def Percentile(self, percentage):
        """
        Computes a percentile of a given Pmf. 计算分位数
        Note: this is not super efficient. If you are planning to compute more than a few percentiles,
        compute the Cdf.
        :param percentage:
        :return: valu from the Pmf.
        """
        p = percentage / 100.0
        total = 0
        for val, prob in sorted(self.Items()):
            total += prob
            if total >= p:
                return val

    def ProbGreater(self, x):
        """
        Probability that a sample from this Pmf exceeds x.
        :param x: number
        :return: float probability.
        """
        if isinstance(x, _DictWrapper):
            return PmfProbGreater(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)

    def ProbLess(self, x):
        """
        Probability that a sample from this Pmf is less than x
        :param x: number
        :return: float probability
        """
        if isinstance(x, _DictWrapper):
            return PmfProbLess(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)

    def __lt__(self, obj):
        """
        Less than. 相当于重载了<
        :param obj: number or _DictWrapper
        :return: float probability
        """
        return self.ProbLess(obj)

    def __gt__(self, obj):
        """
        Greater than. 相当于重载了>
        :param obj: number or _DictWrapper
        :return: float probability
        """
        return self.ProbGreater(obj)

    def __ge__(self, obj):
        """
        Greater than or equal 相当于重载了>=
        :param obj: number or _DictWrapper
        :return: float probability
        """
        return 1 - (self < obj)

    def  __le__(self, obj):
        """
        Less than or equal.
        :param obj: number or _DictWrapper
        :return: float probability
        """
        return 1 - (self > obj)

    def Normalize(self, fraction=1.0):
        """
        Normalizes this PMF so the sum of all probs is fraction.
        :param fraction: what the total should be after normalization.
        :return: the total probability before normalizing.
        """
        print("子类中的Normalize方法")
        if self.log:
            raise ValueError("Normalize: Pmf is under a log transform")

        total = self.Total()
        if total == 0.0:
            raise ValueError('Normalize: total probability is zero.')
            # logging.wraning('Normalize: total probability is zero.')
            # return total.

        factor = fraction/total
        for x in self.d:
            self.d[x] *= factor

        return total

    def Random(self):
        """
        Chooses a random element from this PMF.
        Note：this is not very efficient. If you plan to call this more than a few times,
        consider converting to a CDF.
        :return: float value from the Pmf
        """
        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        raise ValueError('Random: Pmf might not be normalized')

    def Mean(self):
        """
        Computes the mean of a PMF.
        :return: float mean
        """
        mean = 0.0
        for x, p in self.d.items():
            mean += p * x
        return mean

    def Var(self, mu=None):
        """
        Computes the variance of a PMF.
        :param mu: the point around which the variance is computed; if omitted, computes the mean
        :return: float variance
        """
        if mu is None:
            mu = self.Mean()

        var = 0.0
        for x, p in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def Std(self, mu=None):
        """
        Computes the standard deviation of a PMF.
        :param mu: the point around which the variance is computed; if omitted, computes the mean.
        :return: float standard deviation
        """
        var = self.Var(mu)
        return math.sqrt(var)

    def MaximumLikelihood(self):
        """
        Returns the value with the highest probability.
        :return: float probability.
        """
        _, val = max((prob, val) for val, prob in self.Items())
        return val

    def CredibleInterval(self, percentage=90):
        """
        Computes the central credible interval.
        If percentage=90, computes the 90% CI.
        :param percentage: float between 0 and 100.
        :return: sequence of two floats, low and high.
        """
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)

    def __add__(self, ohter):
        """
        Computes the Pmf of the sum of values drawn from self and other.
        :param ohter: anoher Pmf or a scalar
        :return: new Pmf
        """
        try:
            return self.AddPmf(ohter)
        except AttributeError:
            return self.AddConstant(ohter)

    def AddPmf(self, other):
        """
        Computes the Pmf of the sum of values drawn from self and other.
        :param other: another Pmf
        :return: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Itmes():
                pmf.Incr(v1 + v2, p1 * p2)
        return pmf

    def AddConstant(self, other):
        """
        Computes the Pmf of the sum a constant an value from self.
        :param other: a number.
        :return: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """
        Compute the Pmf of the diff of values drawn from self and other.
        :param other: another Pmf.
        :return: new Pmf.
        """
        try:
            return self.SubPmf(other)
        except AttributeError:
            return self.AddConstant(-other)

    def SubPmf(self, other):
        """
        Compute the Pmf of the diff of values drawn froma self and other.
        :param other: another Pmf.
        :return: new Pmf.
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf

    def __mul__(self, other):
        """
        Compute the Pmf ot the product of values drawn from self and other.
        :param other: another Pmf.
        :return: new Pmf.
        """
        try:
            return self.MulPmf(other)
        except AttributeError:
            return self.MulConstant(other)

    def MulPmf(self, other):
        """
        Computes the Pmf of the product of values drawn fromo self and other.
        :param other: another Pmf.
        :return: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 * v2, p1 * p2)
        return pmf

    def MulConstant(self, other):
        """
        Computes the Pmf of the product of values drawn from self and a constant.
        :param other:a number
        :return: new pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1*other, p1)
        return pmf

    def __div__(self, other):
        """
        Computes the Pmf of ratio of values drawn from self and other.
        :param other: another Pmf
        :return: new Pmf
        """
        try:
            return self.DivPmf(other)
        except AttributeError:
            return self.MulConstant(1.0/other)

    __truediv__ = __div__

    def DivPmf(self, other):
        """
        Computes the Pmf of the ratio of values drawn from self and other.
        :param other: another Pmf
        :return: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2, in self.Items():
                pmf.Incr(v1 / v2, p1 * p2)
        return pmf

    def Max(self, k):
        """
        Computes the CDF of the maximum of k selections from this dist.
        :param k:
        :return: new Cdf
        """
        cdf = self.MakeCdf()
        return cdf.Max(k)


def PmfProbLess(pmf1, pmf2):
    """
    Probablity that a value from pmf1 is less than a value from pmf2
    :param pmf1: Pmf object
    :param pmf2: Pmf object
    :return: float probability
    """
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 < v2:
                total += p1 * p2
    return total


def PmfProbGreater(pmf1, pmf2):
    """
    Probability that a value from pmf1 is less than a value from pmf2.
    :param pmf1: Pmf object
    :param pmf2: Pmf object
    :return: float probability
    """
    total = 0.0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 > v2:
                total += p1 * p2
    return total


class Test(object):
    pass

class Test1(object):
    pass


def RandomSeed(x):
    """
    Initialize the random and np.random generators.
    :param x: int seed
    """
    random.seed(x)
    np.random.seed(x)


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""
