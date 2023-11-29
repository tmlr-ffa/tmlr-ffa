#!/usr/bin/env python
#-*- coding:utf-8 -*-

#
#==============================================================================
from __future__ import print_function
import statistics
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
import resource
import csv
import random
import lime
import lime.lime_tabular
import shap
#from anchor import utils
#from anchor import anchor_tabular

#
#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, global_model_name, appr, X_train, y_train, model):
        self.global_model_name = global_model_name
        self.appr = appr
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.init_explainer(appr)

    def init_explainer(self, appr):
        if appr.lower() == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                               #feature_names=self.X_train.columns,
                                                               discretize_continuous=False)
        elif appr.lower() == 'shap':
            self.explainer = shap.Explainer(self.model, self.X_train)

        elif appr.lower() == 'anchor':
            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=[False, True],
                feature_names=self.X_train.columns,
                train_data=self.X_train.values,
                categorical_names={})
        else:
            print('Wrong approach input')
            exit(1)

    def explain(self, X, y):
        pred = self.model.predict(X)[0]

        inst = X.iloc[0]
        preamble = []
        for fid, f in enumerate(inst.index):
            preamble.append(f'{f} = {inst[fid]}')

        #print('\n  Explaining: IF {} THEN defect = {}'.format(' AND '.join(preamble), pred))
        print('  explaining: IF {} THEN defect = {}'.format(' AND '.join(preamble), pred))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.appr.lower() == 'lime':
            self.lime_explain(X, y, pred)
        elif self.appr.lower() == 'shap':
            self.shap_explain(X, y, pred)
        elif self.appr.lower() == 'anchor':
            self.anchor_explain(X, y, pred)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
        
        print('  expl:', sorted(self.lit2imprt.keys(), key=lambda l: abs(l)))
        print('  lit2imprt:', self.lit2imprt)
        print('  time: {0}'.format(self.time))

    def lime_explain(self, X, y, pred):
        #predict_fn = lambda x: self.model.predict_proba(x).astype(float)
        explanation = self.explainer.explain_instance(X.iloc[0, :],
                                          self.model.predict_proba,
                                          num_features=X.shape[1], #10 is the default value
                                          top_labels=1)

        prob0, prob1 = self.model.predict_proba(X)[0]
        pred = 0 if prob0 > prob1 else 1
        exp = explanation.local_exp[pred]
        fid2imprt = {fid: imprt for fid, imprt in exp}
        lit2imprt = {}
        X = X.to_numpy()
        for fid, x in enumerate(X[0]):
            imprt = fid2imprt[fid]
            if imprt != 0:
                lit2imprt[fid] = imprt
        self.lit2imprt = lit2imprt

    def shap_explain(self, X, y, pred):
        shap_values = self.explainer.shap_values(X)
        if len(shap_values) <= 2:
            shap_values = shap_values[-1]
        else:
            shap_values = shap_values[pred][0]
        lit2imprt = {}
        for fid, (x, sp) in enumerate(zip(X.to_numpy()[0], shap_values)):
            lit = fid
            if sp != 0:
                lit2imprt[lit] = sp
        self.lit2imprt = lit2imprt

    def anchor_explain(self, X, y, pred):
        exp = self.explainer.explain_instance(X.values[0], self.model.predict, threshold=0.95)

        # explanation
        expl = [name for f, name in sorted(zip(exp.features(), exp.names()))]

        preamble = ' AND '.join(expl)

        print('  expl: IF {0} THEN defect = {1}'.format(preamble, pred))
        print('  size:', len(expl))
        #print('  Anchor: %s' % (' AND '.join(exp.names())))
        #print('  Precision: %.2f' % exp.precision())
        #print('  Coverage: %.2f' % exp.coverage())


if __name__ == '__main__':
    proj_name = sys.argv[1]
    global_model_name = sys.argv[2]
    appr = sys.argv[3]
    nof_insts = int(sys.argv[4])
    #batch = int(sys.argv[-1])
    #print('batch:', batch)
    #print('Computing explanations using', appr)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    k2name = {'X_train': '../datasets/jit/{}_X_train.csv'.format(proj_name),
              'y_train': '../datasets/jit/{}_y_train.csv'.format(proj_name),
              'X_test': '../datasets/jit/{}_X_test.csv'.format(proj_name),
              'y_test': '../datasets/jit/{}_y_test.csv'.format(proj_name)}

    path_X_train = '../datasets/jit/{}_X_train.csv'.format(proj_name)
    path_y_train = '../datasets/jit/{}_y_train.csv'.format(proj_name)
    X_train = pd.read_csv(path_X_train)
    y_train = pd.read_csv(path_y_train).iloc[:, 0]
    indep = X_train.columns
    dep = 'defect'

    path_X_explain = '../datasets/jit/{}_X_test.csv'.format(proj_name)
    path_y_explain = '../datasets/jit/{}_y_test.csv'.format(proj_name)
    X_explain = pd.read_csv(path_X_explain)
    y_explain = pd.read_csv(path_y_explain).iloc[:, 0]

    path_model = './jit/models/{}_LR_global_model.pkl'.format(proj_name)

    with open(path_model, 'rb') as f:
        model = pickle.load(f)
        
    explainer = HExplainer(global_model_name, appr, X_train, y_train, model)

    """
    
    Explaining
    
    """

    selected_ids = set(range(len(X_explain)))
    if len(X_explain) > nof_insts:
        random.seed(1000)
        selected_ids = set(random.sample(range(len(X_explain)), nof_insts))
    selected_ids = sorted(selected_ids)

    times = []
    nof_inst = 0

    preds = model.predict(X_explain)

    for i in selected_ids:
        print('\ninst:', nof_inst)
        nof_inst += 1

        if i < len(X_explain) - 1:
            X = X_explain.iloc[i: i+1,]
            y = y_explain.iloc[i: i+1,]
        else:
            X = X_explain.iloc[i: , ]
            y = y_explain.iloc[i: , ]

        explainer.explain(X, y)
        times.append(explainer.time)
    
    #print(f'times: {times}\n')
    print()
    print('# of insts:', nof_inst)
    print(f'tot time: {sum(times)}')
    
    exit()
