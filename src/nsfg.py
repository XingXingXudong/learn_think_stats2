# coding: utf-8

import src.thinkstats2 as thinkstats2


def read_fem_preg(dct_file='../data/2002FemPreg.dct', dat_file='../data/2002FemPreg.dat.gz'):
    """
    Reads the NSFG pregnancy data.
    :param dct_file: string file name
    :param dat_file: string file name
    :return: DateFrame
    """
    dct = thinkstats2.read_stata_dct(dct_file)
    df = dct.read_fixd_width(dat_file)
    return df

if __name__ == "__main__":
    data = read_fem_preg()
    print(data.head())

