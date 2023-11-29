#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
##

#
#==============================================================================
from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import KFold


def categorical_features(file):
    df = pd.read_csv(file)
    cat_features = []
    for i, feature in enumerate(df.columns[:-1]):
        if (df[feature].nunique() <= 2) or \
            ( (not np.issubdtype(df[feature].dtypes, int)) and (not np.issubdtype(df[feature].dtypes, float)) ):
            cat_features.append('{0}'.format(i))
    unique_classes = sorted(df[df.columns[-1]].unique())
    xgb_unique_classes = sorted(range(len(unique_classes)))

    if len(cat_features) > 0 or \
        ((not np.issubdtype(df[df.columns[-1]].dtypes, int)) and (not np.issubdtype(df[df.columns[-1]].dtypes, float))) or \
            unique_classes != xgb_unique_classes:

        with open(file + '.catcol', 'w') as f:
            f.write('\n'.join(cat_features))

def prepare_dt(dtfiles):
    cmds = []
    for file in dtfiles:
        catcol = file + '.catcol'

        if not os.path.isfile(catcol):
            continue

        pfiles = '{0},{1}'.format(file.rsplit('/', maxsplit=1)[-1],
                                  file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0])

        cmd = 'python ./explain.py -p --pfiles {0} {1}/'.format(pfiles, file.rsplit('/', maxsplit=1)[0])
        cmds.append(cmd)
    return cmds

def train(dtfiles):
    cmds = []
    ts = [25]
    ds = [3]
    for file_ in dtfiles:
        filename = file_.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
        file = file_.replace('/complete/', '/train/').replace('.csv', '_train.csv')
        pfile = file[ : file.rfind('.')] + '_data.csv'
        if os.path.isfile(pfile):
            for t in ts:
                for d in ds:
                    cmd = 'python ./explain.py -o ./btmodels/{}/ -c --testsplit 0 -t -n {} -d {} {}'.format(filename, t, d, pfile)
                    cmds.append(cmd)
        else:
            for t in ts:
                for d in ds:
                    cmd = 'python ./explain.py -o ./btmodels/{}/ --testsplit 0 -t -n {} -d {} {}'.format(filename, t, d, file)
                    cmds.append(cmd)
    return cmds

def split(file, k=5, inccat=False):
    ori_dir = file.strip('/').rsplit('/', maxsplit=1)[0] + '/'
    saved_dir0 = ori_dir.replace('/complete/', '/train/') + '/'
    saved_dir1 = ori_dir.replace('/complete/', '/test/') + '/'
    if not os.path.isdir(saved_dir0):
        os.makedirs(saved_dir0)
    if not os.path.isdir(saved_dir1):
        os.makedirs(saved_dir1)

    filename = file.rsplit('/', maxsplit=1)[-1]

    df = pd.read_csv(file)
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)

    for i, (train_index, test_index) in enumerate(kf.split(df), start=1):
        df_train, df_test = df.iloc[train_index, :], df.iloc[test_index, :]

        if inccat:
            new_file_train = filename.replace('_data.csv', '_train_data.csv')
            new_file_test = filename.replace('_data.csv', '_test_data.csv')
            #new_file_train = filename.replace('_data.csv', '_train{}_data.csv'.format(i))
            #new_file_test = filename.replace('_data.csv', '_test{}_data.csv'.format(i))

            ori_pkl = filename + '.pkl'
            train_pkl = saved_dir0 + ori_pkl.replace('_data.csv', '_train_data.csv')
            test_pkl = saved_dir1 + ori_pkl.replace('_data.csv', '_test_data.csv')
            #train_pkl = saved_dir0 + ori_pkl.replace('_data.csv', '_train{}_data.csv'.format(i))
            #test_pkl = saved_dir1 + ori_pkl.replace('_data.csv', '_test{}_data.csv'.format(i))
            os.system('cp {} {}'.format(ori_dir + ori_pkl, train_pkl))
            os.system('cp {} {}'.format(ori_dir + ori_pkl, test_pkl))
        else:
            #new_file_train = filename.replace('.csv', '_train{}.csv'.format(i))
            #new_file_test = filename.replace('.csv', '_test{}.csv'.format(i))

            new_file_train = filename.replace('.csv', '_train.csv')
            new_file_test = filename.replace('.csv', '_test.csv')

        df_train.to_csv(saved_dir0 + new_file_train, index=False)
        df_test.to_csv(saved_dir1 + new_file_test, index=False)
        return
#
#==============================================================================
if __name__ == '__main__':

    """
    get datasets
    """
    dtfiles = []
    for root, dirs, files in os.walk('../datasets/tabular/complete'):
        for file in files:
            if (file.endswith('.csv')) and \
                    ('_data.csv' not in file) and\
                    ('_discrete.csv' not in file) and \
                    ('_train' not in file):
                if file.startswith('._'):
                    continue
                dtname = file.rsplit('.', maxsplit=1)[0]
                dtfiles.append(os.path.join(root, file))

    #for file in dtfiles:
    #    categorical_features(file)

    """
    Prepare datasets
    """

    #cmds = prepare_dt(dtfiles)
    #for cmd in cmds:
    #    print(cmd)
    #    os.system(cmd.replace('python3 ', 'python '))
    #    pass

    """
    Split datasets
    """
    k = 5
    for file in dtfiles:
        if os.path.isfile(file.replace('.csv', '_data.csv')):
            inccat = True
            file_ = file.replace('.csv', '_data.csv')
        else:
            inccat = False
            file_ = file
        split(file_, k=k, inccat=inccat)

    """
    train BTs
    """
    #cmds = train(dtfiles)
    #for cmd in cmds:
    #    print(cmd)
    #    #os.system(cmd.replace('python3 ', 'python '))
    #    pass
    exit()
