#!/usr/bin/env python
#-*- coding:utf-8 -*-

#
#==============================================================================
from __future__ import print_function
import statistics
from lrxp import LRExplainer
from train import prepare_data, train_global_model
from options import Options
import os
import sys
import random
import pandas as pd
from dataset import Dataset

#
#==============================================================================
data_path = './dataset/'

if __name__ == '__main__':
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    if options.train:
        x_train, x_test, y_train, y_test = prepare_data(options.proj_name, mode='all')
        train_global_model(options.proj_name, x_train, y_train, options.global_model_name)

    ### explaining
    if options.xtype:
        print('\nExplaining the {0} model...\n'.format('logistic regression' if options.global_model_name.upper() == 'LR' else ''))
        # here are some stats
        nofex, minex, maxex, avgex, times = {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}, \
                                            {xtype: [] for xtype in ['abd', 'con']}

        # Explain data
        data = Dataset(filename='../datasets/jit/'+options.proj_name+'.csv', mapfile=options.mapfile,
                       separator=options.separator, use_categorical=options.use_categorical)
        insts = pd.read_csv(options.inst)

        selected_ids = set(range(len(insts)))
        if options.nof_inst:
            if len(insts) > options.nof_inst:
                random.seed(1000)
                selected_ids = random.sample(range(len(insts)), options.nof_inst)
        selected_ids = sorted(selected_ids)

        nof_inst = 0
        for id in selected_ids:#range(len(insts)):
            #if id not in selected_ids:
            #    continue

            nof_inst += 1
            inst = insts.iloc[id]

            explainer = LRExplainer(data, options)

            expls, time = explainer.explain(inst)
            # need check
            if options.xnum == 1:
                xtypes = ['abd' if options.xtype in ['abd', 'abductive'] else 'con']
            else:
                xtypes = ['abd', 'con']

            for xtype in xtypes:
                nofex[xtype].append(len(expls[xtype]))
                minex[xtype].append(min([len(e) for e in expls[xtype]]))
                maxex[xtype].append(max([len(e) for e in expls[xtype]]))
                avgex[xtype].append(statistics.mean([len(e) for e in expls[xtype]]))

            times['abd'].append(time['abd'])
            times['con'].append(time['con'])

        if options.verb > 0:
            print('')
            print('')
            print('# of insts:', nof_inst)

            xtype = 'abd' if options.xtype in ['abd', 'abductive'] else 'con'
            print('')
            #print('{0} times: {1}\n'.format(xtype, times[xtype]))
            print('tot # of {0} expls: {1}'.format(xtype, sum(nofex[xtype])))
            print('min # of {0} expls: {1}'.format(xtype, min(nofex[xtype])))
            print('avg # of {0} expls: {1:.2f}'.format(xtype, statistics.mean(nofex[xtype])))
            print('max # of {0} expls: {1}'.format(xtype, max(nofex[xtype])))
            print('')
            print('Min {0} expl sz: {1}'.format(xtype, min(minex[xtype])))
            print('min {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(minex[xtype])))
            print('avg {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(avgex[xtype])))
            print('max {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(maxex[xtype])))
            print('Max {0} expl sz: {1}'.format(xtype, max(maxex[xtype])))
            print('')
            print('tot {0} time: {1:.2f}'.format(xtype, sum(times[xtype])))
            print('min {0} time: {1:.2f}'.format(xtype, min(times[xtype])))
            print('avg {0} time: {1:.2f}'.format(xtype, statistics.mean(times[xtype])))
            print('max {0} time: {1:.2f}'.format(xtype, max(times[xtype])))

            if options.xnum != 1:
                xtype = 'con' if options.xtype in ['abd', 'abductive'] else 'abd'
                print('')
                #print('{0} times: {1}\n'.format('abd' if xtype == 'abd' else 'con', times[xtype]))
                print('tot # of {0} expls: {1}'.format(xtype, sum(nofex[xtype])))
                print('min # of {0} expls: {1}'.format(xtype, min(nofex[xtype])))
                print('avg # of {0} expls: {1:.2f}'.format(xtype, statistics.mean(nofex[xtype])))
                print('max # of {0} expls: {1}'.format(xtype, max(nofex[xtype])))
                print('')
                print('Min {0} expl sz: {1}'.format(xtype, min(minex[xtype])))
                print('min {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(minex[xtype])))
                print('avg {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(avgex[xtype])))
                print('max {0} expl sz: {1:.2f}'.format(xtype, statistics.mean(maxex[xtype])))
                print('Max {0} expl sz: {1}'.format(xtype, max(maxex[xtype])))
                print('')
                print('tot {0} time: {1:.2f}'.format(xtype, sum(times[xtype])))
                print('min {0} time: {1:.2f}'.format(xtype, min(times[xtype])))
                print('avg {0} time: {1:.2f}'.format(xtype, statistics.mean(times[xtype])))
                print('max {0} time: {1:.2f}'.format(xtype, max(times[xtype])))

            '''
            print('')
            print('tot time: {0:.2f}'.format(sum(times['abd']) + sum(times['con'])))
            print('min time: {0:.2f}'.format(sum(times['abd']) + sum(times['con'])))
            print('avg time: {0:.2f}'.format(sum(times['abd']) + sum(times['con'])))
            print('max time: {0:.2f}'.format(sum(times['abd']) + sum(times['con'])))
            '''

    exit()





