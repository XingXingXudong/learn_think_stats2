#conding: utf-8


"""
Chapter4 Cumulative Distribution functions.
"""

import src.thinkstats2 as thinkstats2


def MakeExample():
    """Makes a simple example CDF."""
    t = [2, 1, 3, 2, 5]
    cdf = thinkstats2.Cdf(t)



def main(name, data_dir=''):
    thinkstats2.RandomSeed(17)

    MakeExample()


if __name__ == '__main__':
    import sys
    main(*sys.argv)

