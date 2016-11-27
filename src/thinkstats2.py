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






