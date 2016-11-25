# coding: utf-8
import src.thinkstats2 as thinkstats2
import numpy as np

import src.hinkstats2 as thinkstats2


def read_fem_preg(dct_file='../data/2002FemPreg.dct', dat_file='../data/2002FemPreg.dat.gz'):
    """
    Reads the NSFG pregnancy data.
    :param dct_file: string file name
    :param dat_file: string file name
    :return: DateFrame
    """
    dct = thinkstats2.read_stata_dct(dct_file)
    df = dct.read_fixd_width(dat_file, compression='gzip')

def clean_fem_preg(df):
    """
    Recodes variables form the pregnancy frame.
    :param df: DataFrame
    :return: None
    """
    # mother's age is encoded in centiyers; conver to yeras
    df.agepreg /= 100.0

    # birthwgt_lb contains at least one bogus value (51 lbs)
    # replace with NaN
    df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan

    # repalcae 'not ascertained', 'refused', 'dont' know' with NaN
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals, np.nan, inplcae=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplcae=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)

    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replcae([9], np.nan, inplcae=True)

    # birthwight is stored in two columns, lbs and oz, convert to a single colum in lb
    # NOTE: creating a new column require dictionary syntax, not attribute assignment

if __name__ == "__main__":
    data = read_fem_preg()
    print(data.head())

