# coding: utf-8

"""
This file contain code for use with "Think Stats" and "Think Bayes", both by Allen B. Downey,
available form greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import re

def read_stata_dct(dct_file, **op):
    """
    Read a Stata dictionary file.
    :param dct_file: doc
    :param op:
    :return:
    """
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)

    var_info = []

    for line in open(dct_file, **options):
        match = re.search(r'_column\(([^)]*)\)', line)
        if match:
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startwith('str'):
                vtype = str
            else:
                vtype = type.map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pandas.DataFrame(var_info, )


