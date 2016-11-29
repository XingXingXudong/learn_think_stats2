# coding: utf-8

"""
Chapter 3
"""

import src.first as first
import src.thinkstats2 as thinkstats2


def MakeFigures(firsts, ohters):
    """
    Plot Pmfs of pregnancy length.
    :param firsts: DataFrame
    :param ohters: DataFrame
    """
    # plot th PMFs
    first_pmf = thinkstats2.Pmf(first.prglngth, label='first')


def main(script):
    live, firsts, others = first.MakeFrames()



if __name__ == "__main__":
    import sys
    main(*sys.argv)