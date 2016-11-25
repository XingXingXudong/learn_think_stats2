# coding: utf-8

import numpy as np
from collections import defaultdict

import src.thinkstats2 as thinkstats2
import sys


def read_fem_preg(dct_file='../data/2002FemPreg.dct', dat_file='../data/2002FemPreg.dat.gz'):
    """
    Reads the NSFG pregnancy data.
    :param dct_file: string file name
    :param dat_file: string file name
    :return: DateFrame
    """
    dct = thinkstats2.read_stata_dct(dct_file)
    df = dct.read_fixd_width(dat_file, compression='gzip')
    clean_fem_preg(df)
    return df


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
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)

    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replace([9], np.nan, inplace=True)

    # birthwight is stored in two columns, lbs and oz, convert to a single colum in lb
    # NOTE: creating a new column require dictionary syntax, not attribute assignment
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    # due to a bug in read_stat_dct, the last variable gets clipped;
    # so for now set it to NaN
    df.cmintvw = np.nan


def make_pre_map(df):
    """
    Make a map from caseid to list of preg indices.
    :param df: DataFrame
    :return: dict that maps from caeid ot list of indices into preg df
    """
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d


def main(script):
    """
    Tests the functions in this module.
    :param script: string script name
    :return: None
    """
    df = read_fem_preg()
    print(df.head())
    print(df.shape)

    assert len(df) == 13593
    assert df.caseid[13592] == 12571
    assert df.pregordr.value_counts()[1] == 5033
    print(df.pregordr.value_counts())
    assert df.nbrnaliv.value_counts()[1] == 8981
    assert df.babysex.value_counts()[1] == 4641
    assert df.birthwgt_lb.value_counts()[7] == 3049
    assert df.birthwgt_oz.value_counts()[0] == 1037
    assert df.prglngth.value_counts()[39] == 4744
    assert df.outcome.value_counts()[1] == 9148
    assert df.birthord.value_counts()[1] == 4413
    assert df.agepreg.value_counts()[22.75] == 100
    assert df.totalwgt_lb.value_counts()[7.5] == 302

    weights = df.finalwgt.value_counts()
    key = max(weights.keys())
    assert df.finalwgt.value_counts()[key] == 6

    print("%s: All tests passed." % script)


if __name__ == "__main__":
    main(*sys.argv)
