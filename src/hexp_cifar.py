#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## hexp.py
##

# imported modules:
#==============================================================================
from __future__ import print_function
import collections
import resource
import sys
import os
from copy import copy, deepcopy
import torch.nn as nn
import torch
import argparse
import random
from tqdm import tqdm
import numpy as np
import pickle
import torchvision
from pysat.formula import IDPool, WCNF, CNF
import json
import statistics
import matplotlib.pyplot as plt
import time
import resource
import csv
import random
import lime
from lime import lime_image
from lime import lime_tabular
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap
from itertools import chain
import math
import collections
import torch.nn.functional as F
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import getopt
import pandas as pd
from xgbooster import XGBooster, preprocess_dataset, discretize_dataset
from options import Options
from skimage.color import gray2rgb, rgb2gray, label2rgb
import functools
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
#from fastshap.utils import MaskLayer1d
#from fastshap import Surrogate, KLDivLoss
#from fastshap import FastSHAP
#from shapreg import removal, games, shapley

#==============================================================================

class HExplainer(object):
    #HeuristicExplainer
    def __init__(self, args, xgb, X_train, X_ori_train, Y_train, Y_ori_train, target_names, info):
        if args.appr.lower() not in ('lime', 'shap', 'anchor', 'fastshap',
                                     'shapreg', 'kernelshap'):
            print('wrong approach')
            exit(1)
        random.seed(1234)
        self.X_ori_train = X_ori_train
        self.Y_ori_train = Y_ori_train
        self.appr = args.appr
        self.xgb = xgb
        self.args = args
        self.info = info
        self.features = X_train.columns
        self.init(args, X_train, X_ori_train, target_names)

    def init(self, args, X_train, X_ori_train, target_names):
        if not info:
            categorical_features = []
        else:
            categorical_features = self.info['categorical_features']
        #categorical_features = range(len(X_train.columns))
        if args.appr.lower() == 'lime':
            #print(X_train.loc[:, X_train.columns])
            #X_train = X_train.astype(str)
            self.explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
                                                               feature_names=X_train.columns,
                                                               class_names=target_names,
                                                               categorical_features=categorical_features,
                                                               discretize_continuous=False)\
                if args.tabular else lime_image.LimeImageExplainer()

        elif args.appr.lower() == 'shap':
            if self.args.tabular:
                self.explainer = shap.TreeExplainer(self.xgb.model)
            else:
                self.explainer = None
        elif args.appr.lower() == 'kernelshap':
            self.explainer = shap.KernelExplainer(self.predict_proba, self.X_ori_train, link='logit')
        elif args.appr.lower() == 'fastshap':
            self.explainer = self.prepare_fastshap()
        elif args.appr.lower() == 'shapreg':
            self.explainer = self.prepare_shapreg()
        else:
            # self.appr.lower() == 'anchor'
            if self.args.tabular:
                self.explainer = anchor_tabular.AnchorTabularExplainer(class_names=target_names,
                                                                       feature_names=X_train.columns,
                                                                       train_data=X_train.to_numpy(),
                                                                       categorical_names=dict(enumerate(X_train.columns)))
            else:
                def to_3c(x):
                    x0 = self.flat2matrix(x)
                    x1 = np.expand_dims(x0, axis=2)
                    x2 = np.repeat(x1, 3, axis=2)
                    return x2
                X_ori_train_3c = np.applY_along_axis(to_3c, 1, X_ori_train)
                self.explainer = anchor.anchor_image.AnchorImage(dummys=X_ori_train_3c,
                                                                 n=5000)
            #def transform_img_fn(binarY_train):
            #    outs = []
            #    for x in binarY_train:
            #        out = torch.cat([x for i in range(3)], 0)
            #        out = np.transpose(out, (1, 2, 0))

            #        out = out[None, :]
            #        outs.append(out)
            #    outs = torch.cat(outs, 0).detach().cpu().numpy()
            #    return outs

            #train_dataloaders, testset, nof_classes = load_data(args, train=True)
            #train = torch.cat([v[0] for v in train_dataloaders["val"]])
            #binarY_train = self.preprocessing(train).to(args.device)

            #outs = transform_img_fn(binarY_train)
            #self.explainer = anchor.anchor_image.AnchorImage(dummys=outs,
            #                                                 n=5000)

    def prepare_fastshap(self):
        device = torch.device('cpu')
        X_ori_train, X_ori_val, Y_ori_train, Y_ori_val = train_test_split(
            self.X_ori_train, self.Y_ori_train, test_size=0.2, random_state=0)
        X_ori_train = torch.tensor(X_ori_train.values, dtype=torch.float32, device=device)
        X_ori_val = torch.tensor(X_ori_val.values, dtype=torch.float32, device=device)
        num_features = X_ori_train.shape[1]

        def create_surrogate():
            saved_dir = './fastshap_models/sur_models/'
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)
            saved_file = saved_dir + self.args.test.rsplit('datasets/', maxsplit=1)[-1].replace('/', '_') + '_tab.pt'
            if os.path.isfile(saved_file):
                surr = torch.load(saved_file).to(device)
                surrogate = Surrogate(surr, num_features)
            else:
                # Create surrogate model
                surr = nn.Sequential(
                    MaskLayer1d(value=0, append=True),
                    nn.Linear(2 * num_features, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, 2)).to(device)

                # Set up surrogate object
                surrogate = Surrogate(surr, num_features)

                # Set up original model
                def original_model(x):
                    pred = self.predict_proba(x.cpu().numpy())
                    if pred.shape[1] > 2:
                        for dim_id in range(2, pred.shape[1]):
                            pred[:, 0] += pred[:, 2]
                        pred = pred[:, :2]
                    return torch.tensor(pred, dtype=torch.float32, device=x.device)

                # Train
                surrogate.train_original_model(
                    X_ori_train,
                    X_ori_val,
                    original_model,
                    batch_size=64,
                    max_epochs=100,
                    loss_fn=KLDivLoss(),
                    validation_samples=10,
                    validation_batch_size=10000,
                    verbose=True)
                # Save surrogate
                surr.cpu()
                torch.save(surr, saved_file)
                surr.to(device)
            return surrogate

        def create_fastshap():

            saved_dir = './fastshap_models/fastshap_explainers/'
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)
            saved_file = saved_dir + self.args.test.rsplit('datasets/', maxsplit=1)[-1].replace('/',
                                                                           '_') + '_tab.pt'
            if os.path.isfile(saved_file):
                explainer = torch.load(saved_file).to(device)
                fastshap = FastSHAP(explainer, surrogate, normalization='additive',
                                    link=nn.Softmax(dim=-1))
            else:
                # Create explainer model
                explainer = nn.Sequential(
                    nn.Linear(num_features, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2 * num_features)).to(device)

                # Set up FastSHAP object
                fastshap = FastSHAP(explainer, surrogate, normalization='additive',
                                    link=nn.Softmax(dim=-1))

                # Train
                fastshap.train(
                    X_ori_train,
                    X_ori_val, #[:100],
                    batch_size=32,
                    num_samples=32,
                    max_epochs=200,  # 200,
                    validation_samples=128,
                    verbose=True)

                # Save explainer
                explainer.cpu()
                torch.save(explainer, saved_file)
                explainer.to(device)
            return fastshap

        surrogate = create_surrogate()
        fastshap = create_fastshap()
        return fastshap

    def prepare_shapreg(self):
        X_ori_train, X_ori_val, Y_ori_train, Y_ori_val = train_test_split(
            self.X_ori_train, self.Y_ori_train, test_size=0.2, random_state=0)

        # Make model callable
        def model_predict_prob(x):
            try:
                pred = self.predict_proba(x.values)
            except:
                pred = self.predict_proba(x)
            if pred.shape[1] == 2:
                pred = np.squeeze(pred[:, 1])
            return pred

        model_lam = lambda x: model_predict_prob(x)

        # Model extension
        marginal_extension = removal.MarginalExtension(X_ori_val.values[:512], model_lam)
        return marginal_extension

    def explain(self, X, X_ori, inst_id):
        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime

        if self.args.verbose:
            preamble = []
            for i, (f, v) in enumerate(zip(self.features, X[0])):
                if self.info and i in self.info['categorical_names']:
                    preamble.append('{} = {}'.format(f, info['categorical_names'][i][int(v)]))
                else:
                    preamble.append('{} = {}'.format(f, v))
            print('  explaining:', ' AND '.join(preamble))

        if self.appr.lower() == 'lime':
            lit2imprt, pred = self.lime_explain(X, X_ori)
        elif self.appr.lower() == 'shap':
            lit2imprt, pred = self.shap_explain(X, X_ori)
        elif self.appr.lower() == 'kernelshap':
            lit2imprt, pred = self.kernelshap_explain(X, X_ori)
        elif self.appr.lower() == 'fastshap':
            lit2imprt, pred = self.fastshap_explain(X, X_ori)
        elif self.appr.lower() == 'shapreg':
            lit2imprt, pred = self.shapreg_explain(X, X_ori)
        else:
            assert self.appr.lower() == 'anchor'
            lit2imprt, pred = self.anchor_explain(X, X_ori)

        #if self.args.visual and 'mnist' in self.args.test.lower():
        #    self.visualise(lit2imprt, inst_id, pred, shape=self.args.shape)

        time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
               resource.getrusage(resource.RUSAGE_SELF).ru_utime - time

        if ('mnist' not in self.args.test.lower()) and ('cifar' not in self.args.test.lower()):
            lit2imprt = {abs(lit)-1: imprt for lit, imprt in lit2imprt.items()}
        elif 'origin' in self.args.test or 'ori' in self.args.test or 'cifar' in self.args.test:
            lit2imprt = {abs(lit): imprt for lit, imprt in lit2imprt.items()}

            if 'cifar' in self.args.test.lower():
                if '32,32' in args.test:
                    pix2imprt = collections.defaultdict(lambda : 0)
                    for lit in lit2imprt:
                        pixel = (abs(lit)-1) % (32 * 32) + 1
                        pix2imprt[pixel] += lit2imprt[lit]
                    lit2imprt = pix2imprt

        #image start from index 1

        if self.args.verbose:
            print('  expl:', sorted(lit2imprt.keys(), key=lambda l: abs(l)))
            print('  lit2imprt:', lit2imprt)
            print('  time:', time)

    def lime_explain(self, X, X_ori):
        """
        explaining images using LIME
        """

        if self.args.tabular:
            pred = self.predict(X[0])[0]
            explanation = self.explainer.explain_instance(X[0],
                                                  self.predict_proba,
                                                  num_features=X.shape[1],
                                                  top_labels=1)

            exp = explanation.local_exp[pred]
            var2imprt = {var+1: imprt for var, imprt in exp}
            lit2imprt = {}
            for var, x in enumerate(X[0], start=1):
                lit = var if x > 0 else -var
                imprt = var2imprt[var]
                if imprt != 0:
                    lit2imprt[lit] = imprt
        else:
            if self.args.segalg == 'quickshift':
                segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
            else:
                segmenter = SegmentationAlgorithm('slic', n_segments=self.args.nof_seg)

            #X = self.flat2matrix(X[0])
            #X = np.expand_dims(X, axis=2)
            #X = np.repeat(X, 3, axis=2)
            #pred = self.predict(X)
            #print('X:', X)

            X_ori = self.flat2matrix(X_ori[0])
            X_ori = np.expand_dims(X_ori, axis=2)
            X_ori = np.repeat(X_ori, 3, axis=2)
            pred = self.predict(X_ori)[0]
            #print('X_ori:', X_ori)

            explanation = self.explainer.explain_instance(X_ori,
                                                          classifier_fn=self.predict_proba,
                                                          top_labels=10, hide_color=0,
                                                          num_samples=1000,#, #10000
                                                          segmentation_fn=segmenter)
            exp = explanation.local_exp[pred]

            seg_min = np.min(explanation.segments)
            if seg_min > 0:
                explanation.segments = explanation.segments - seg_min
            segments = explanation.segments

            seg2imprt = {seg: imprt for seg, imprt in exp}
            lit2imprt = {}
            flat_X = X[0, :]
            flat_seg = segments.flatten()
            for i, (x, seg) in enumerate(zip(flat_X, flat_seg), start=1):
                imprt = seg2imprt[seg]
                lit = i if x > 0 else -i
                lit2imprt[lit] = imprt

            #print(lit2imprt)
            #exit()
            #print('flat_X.shape:', flat_X.shape)
            #print(flat_seg.shape)
            #exit()
            #print('X.shape:', X.shape)
            #print('segments.shape:', segments.shape)
            ## for x, seg in zip()
            #print('seg2imprt:', seg2imprt)
            #print('exp:', exp)
            #print('segments:')
            #print(explanation.segments)
            #exit()

            #num_features = functools.reduce(lambda x, y: x * y, self.args.shape)
            #hide_rest = False
            #min_weight = 0.01
            #temp, mask = explanation.get_image_and_mask(pred, positive_only=True, num_features=num_features,
            #                                            hide_rest=hide_rest, min_weight=min_weight)


            #print('temp:', temp)
            #print()
            #print('mask:', mask)
            #print()
            #exp = explanation.local_exp[pred]
            #print('exp:', exp)
            #print(' explanation.local_exp[1]:',  explanation.local_exp[(pred + 1 ) % 2])
            #print()
            #print()
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            #ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
            #ax1.set_title('Positive Regions for {}'.format(pred))
            ## temp is the target image
            #temp, mask = explanation.get_image_and_mask(pred, positive_only=False, num_features=num_features,
            #                                            hide_rest=hide_rest, min_weight=min_weight)
            #print('temp:', temp)
            #print()
            #print('mask:', mask)
            #ax2.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
            #ax2.set_title('Positive/Negative Regions for {}'.format(pred))
            #fig.savefig('ab.pdf')

        return lit2imprt, pred
        #test = np.transpose(ori_binarY_test, (1, 2, 0))
        #test = test * 255
        #test = torch.cat([test for i in range(3)], 2)
        #test = test.detach().cpu().numpy()
        #self.lime_predict(test)

        #num_features = 100000
        #expl = self.explainer.explain_instance(test,
        #                                       self.lime_predict,  # classification function
        #                                       top_labels=1,
        #                                       hide_color=-1,
        #                                       num_samples=1000)#1000)
        #temp, mask = expl.get_image_and_mask(expl.top_labels[0], positive_only=True,
        #                                     num_features=num_features, hide_rest=False)
        #print(temp)
        #print()
        #print(mask)
        ##plt.axis("off")
        ### plt.margins(0, 0)
        ##plt_image = plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ### plt.savefig(file.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0)
        ##plt.savefig('lime_{}.pdf'.format(inst_id), bbox_inches='tight', pad_inches=0)
        ##plt.close()
        #self.expl = mask

    def shap_explain(self, X, X_ori):
        """
        explaining images using SHAP
        """
        if self.args.tabular:
            pred = self.predict(X)[0]
            flat_sample = self.trans2btsamps(X)
            shap_values = self.explainer.shap_values(flat_sample)

            if len(shap_values) <= 2:
                shap_values_sample = shap_values[-1]
            else:
                shap_values_sample = shap_values[pred][0]

            sum_values = []
            if (self.xgb.use_categorical):
                p = 0
                for f in range(len(self.features)):
                    if f in self.xgb.categorical_features:
                        nb_values = len(self.xgb.categorical_names[f])
                        sum_v = 0
                        for i in range(nb_values):
                            sum_v = sum_v + shap_values_sample[p + i]
                        p += nb_values
                        sum_values.append(sum_v)
                    else:
                        sum_values.append(shap_values_sample[p])
                        p += 1
            else:
                sum_values = shap_values_sample

            #todo
            #if pred == 0:
            #    sum_values = [-sp for sp in sum_values]

            lit2imprt = {}
            for var, (x, sp) in enumerate(zip(X[0], sum_values), start=1):
                lit = var if x > 0 else -var
                if sp != 0:
                    lit2imprt[lit] = sp

            return lit2imprt, pred
        else:
            self.test_3c = np.transpose(torch.cat([X_ori for i in range(3)], 0),
                                        (1, 2, 0))
            self.segments_slic = slic(self.test_3c, n_segments=self.args.segment,
                                      compactness=self.args.compactness, sigma=self.args.sigma)

            bun2vars = collections.defaultdict(lambda : [])

            for x in range(self.segments_slic.shape[0]):
                for y in range(self.segments_slic.shape[1]):
                    var = x * self.segments_slic.shape[0] + y + 1
                    bun2vars[self.segments_slic[x][y]].append(var)

            nof_buns = len(bun2vars.keys())

            self.explainer = shap.KernelExplainer(self.shap_predict,
                                                  np.zeros((1, nof_buns)))
            test_bundles = np.ones((1, nof_buns))
            pred = np.argmax(self.shap_predict(test_bundles))
            shap_values = self.explainer.shap_values(test_bundles, nsamples=1000)
            try:
                shap_values = shap_values[pred][0]
            except:
                shap_values = shap_values[-1]

            self.expl = np.zeros(self.segments_slic.shape)
            for b, shapv in enumerate(shap_values, start=1):
                #if shapv > 0:
                pixels = bun2vars[b]
                for pvar in pixels:
                    x = (abs(pvar) - 1) // self.segments_slic.shape[-1]
                    y = (abs(pvar) - 1) % self.segments_slic.shape[-1]
                    self.expl[x, y] = 1

    def kernelshap_explain(self, X, X_ori):
        """
        explaining images using SHAP
        """
        kernelshap_values = self.explainer.shap_values(X_ori, nsamples=100)
        pred = self.predict(X_ori)[0]

        if len(kernelshap_values) <= 2:
            shap_values = kernelshap_values[0][-1]
        else:
            shap_values = kernelshap_values[pred][0]

        lit2imprt = {}
        for var, (x, sp) in enumerate(zip(X[0], shap_values), start=1):
            lit = var if x > 0 else -var
            if sp != 0:
                lit2imprt[abs(lit)] = sp

        return lit2imprt, pred

    def fastshap_explain(self, X, X_ori):
        """
        explaining images using SHAP
        """
        pred = self.predict_proba(X_ori)[0]
        for dim_id in range(2, len(pred)):
            pred[1] += pred[dim_id]
        pred = np.argmax(pred)

        fastshap_values = self.explainer.shap_values(X_ori)[0]

        try:
            shap_values = fastshap_values[:, pred]
        except:
            shap_values = fastshap_values[:, -1]

        lit2imprt = {}
        for var, (x, sp) in enumerate(zip(X[0], shap_values), start=1):
            lit = var if x > 0 else -var
            if sp != 0:
                lit2imprt[lit] = sp

        return lit2imprt, pred

    def shapreg_explain(self, X, X_ori):
        """
        explaining images using SHAPREG
        """
        pred = self.predict(X_ori)[0]

        # Set up game (single prediction)
        instance = X_ori[0]
        game = games.PredictionGame(self.explainer, instance)

        # Run estimator
        if 'mnist' in self.args.test:
            shapreg_values = shapley.ShapleyRegression(game, batch_size=32)
        else:
            shapreg_values = shapley.ShapleyRegression(game, batch_size=32)
        try:
            shap_values = shapreg_values.values[:, pred]
        except:
            shap_values = shapreg_values.values

        lit2imprt = {}
        for var, (x, sp) in enumerate(zip(X[0], shap_values), start=1):
            lit = var if x > 0 else -var
            if sp != 0:
                lit2imprt[lit] = sp

        return lit2imprt, pred

    def anchor_explain(self, X, X_ori):
        if self.args.tabular:
            pred = self.predict(X[0])[0]
            explanation = self.explainer.explain_instance(X[0], self.predict, threshold=0.99)

            lit2imprt = {}
            for f_id, prec in zip(explanation.exp_map['feature'],
                                  explanation.exp_map['precision']):
                var = f_id + 1
                x = X[0][f_id]
                lit = var if x > 0 else -var
                if prec > 0:
                    lit2imprt[lit] = prec
            return lit2imprt, pred
        else:
            X_ori = self.flat2matrix(X_ori[0])
            X_ori = np.expand_dims(X_ori, axis=2)
            X_ori = np.repeat(X_ori, 3, axis=2)
            pred = self.predict(X_ori)[0]
            segments, exp = self.explainer.explain_instance(X_ori, self.predict_proba,
                                                            threshold=0.00, batch_size=100,
                                                            tau=0.20, verbose=True,
                                                            min_shared_samples=200, beam_size=2)
            print(segments)
            print(dir(exp))
            exit()
            return
            exit()
            self.expl = segments

    def flat2matrix(self, flat):
        return np.stack(np.split(flat, self.args.shape[0]))

    def binarise(self, pixel):
        return 1 if pixel >= self.args.threshold else 0

    def predict(self, samples):
        scores = self.predict_proba(samples)
        winners = np.argmax(scores, axis=1)
        return winners

    def trans2btsamps(self, samples):
        if self.args.tabular:
            if len(samples.shape) == 1:
                samples = np.expand_dims(samples, axis=0)
            flat_samples = samples.copy()
        else:
            if len(samples.shape) == 3:
                samples = np.expand_dims(samples, axis=0)

            samples = samples[:, :, :, 0]
            flat_samples = samples.reshape(samples.shape[0],
                                           self.args.shape[0] * self.args.shape[1])

            # this step is not needed if it's been binarised
            flat_samples = np.vectorize(self.binarise)(flat_samples)

        flat_samples = self.xgb.transform(flat_samples)
        flat_samples = np.asarray(flat_samples)

        return flat_samples

    def predict_proba(self, samples):
        flat_samples = self.trans2btsamps(samples)
        scores = self.xgb.model.predict_proba(flat_samples)
        return scores

    def shap_predict(self, test_bundles):
        def mask_image(zs, segmentation, image, background=-1):
            out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
            for i in range(zs.shape[0]):
                out[i, :, :, :] = image
                for j in range(zs.shape[1]):
                    if zs[i, j] == 0:
                        var = random.randint(0, 255)
                        #out[i][segmentation == j, :] = background
                        out[i][segmentation == j, :] = 1 if var >= 156 else 0
            return out

        background = -1
        tests = mask_image(test_bundles, self.segments_slic,
                           self.test_3c, background)
        scores = []
        for test_bundle in tests:
            test = torch.tensor(test_bundle)
            test = test[:, :, 0:1]
            test =np.transpose(test, (2, 0, 1))
            # preparing the representation of the instance
            assumps = []
            var = 0
            for c in range(test.shape[-3]):
                for x in range(test.shape[-2]):
                    for y in range(test.shape[-1]):
                        var += 1
                        if test[c, x, y] > 0:
                            assump = var
                            assumps.append(assump)
                        elif test[c, x, y] == 0:
                            assump = -var
                            assumps.append(assump)
                        else:
                            print('skip')
            score = self.cnn_explainer.predict_(assumps)
            scores.append(score)
        scores = torch.tensor(scores)
        probs = F.softmax(scores, dim=1)
        probs = probs.detach().cpu().numpy()
        return probs

    def visualise(self, lit2imprt, inst_id, pred, suffix='', shape=(10, 10), size=11):
        img = np.zeros(shape)
        img = np.expand_dims(img, axis=0)
        img_inst = torch.tensor(img)

        m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}
        count_ = {}
        for lit in lit2imprt:
            x = (abs(lit) - 1) // img_inst.shape[-1]
            y = (abs(lit) - 1) % img_inst.shape[-2]
            imprt = lit2imprt[lit]
            if lit > 0:
                m_3c_instance['R'][0, x, y] = abs(imprt)
                count_[lit] = imprt
            else:
                m_3c_instance['R'][0, x, y] = -abs(imprt)
                count_[lit] = -imprt

        # expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)
        expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R']], 0)
        # print(expl_3c.cpu().numpy().shape)
        a = np.transpose(expl_3c, (1, 2, 0))
        plt.axis("off")
        # orig_cmap = matplotlib.cm.seismic
        # shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shifted')
        # shrunk_cmap = shiftedColorMap(orig_cmap, start=min(var_count.values())/ base, midpoint=0, stop=max(var_count.values())/ base, name='shrunk')
        # imshow = plt.imshow(a, cmap=shrunk_cmap)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
        # print('min(var_count.values():', min(var_count.values()))
        # print('max(var_count.values():', max(var_count.values()))
        # divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min(var_count.values())/ base, vcenter=0,
        #                                         vmax=max(var_count.values())/ base)
        # imshow = plt.imshow(a, cmap='seismic', norm=divnorm)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
        vmax = abs(max(count_.values(), key=lambda l: abs(l))) if len(count_) > 0 else 1
        vmin = -vmax
        imshow = plt.imshow(a, cmap='seismic', vmin=vmin,
                            vmax=vmax)  # ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
        #for lit in lit2imprt:
        #    x = (abs(lit) - 1) // img_inst.shape[-1]
        #    y = (abs(lit) - 1) % img_inst.shape[-2]
        #    plt.text(y, x, '{0:.2f}'.format(lit2imprt[lit]),  # if 'axp' not in suffix else var_count[var],
        #             ha='center', va='center', color='orange', size=size)
        colorbar = plt.colorbar(imshow, location='right')
        # plt.imshow(a)
        saved_dir = '{}/{}'.format(self.args.visual, self.args.appr)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        downscale = '28,28' if '10,10' not in self.args.test else '10,10'
        if 'pneumoniamnist' in self.args.test:
            dtname = 'pneumoniamnist'
        elif '/mnist/' in self.args.test.lower():
            dtname = 'mnist'
        else:
            dtname = self.args.test.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=1)[0]
            downscale = None

        cls = 'all'
        if '1,3' in self.args.test:
            cls = '1,3'
        elif '1,7' in self.args.test:
            cls = '1,7'

        filename = '{}/{}{}_{}{}_{}{}_mix.pdf'.format(saved_dir,
                                                     dtname if 'origin' not in self.args.test else dtname + '_origin',
                                                     '_' + downscale if downscale else '',
                                                       '_' + cls if cls != 'all' else '',
                                               inst_id,
                                               'tab' if self.args.tabular else 'img',
                                                  suffix)
        print('  heatmap saved to', filename)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return filename

    #def visualise(self, expl, batch, inst_id, instance):
    #    if not os.path.isdir(self.args.visual):
    #        os.makedirs(self.args.visual)
    #    print(instance.shape)
    #    m_3c_instance = {colour: instance.clone().detach() for colour in ['R', 'G', 'B']}

    #    instance[0, 0, 2] = 1

    #    for lit in expl:
    #        x = (abs(lit) - 1) // self.expl.shape[0]
    #        y = (abs(lit) - 1) % self.expl.shape[0]
    #        if lit > 0:
    #            m_3c_instance['R'][0, x, y] = 22 / 255
    #            m_3c_instance['G'][0, x, y] = 169 / 255
    #            m_3c_instance['B'][0, x, y] = 191 / 255
    #        else:
    #            m_3c_instance['R'][0, x, y] = 249 / 255
    #            m_3c_instance['G'][0, x, y] = 73 / 255
    #            m_3c_instance['B'][0, x, y] = 82 / 255

    #    expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)

    #    saved_dir = '{0}/hexp/{1}/{2}/{3}'.format(self.args.visual.strip('/'),
    #                                              self.args.appr,
    #                                     self.args.explain,
    #                                     self.args.classes)

    #    if not os.path.isdir(saved_dir):
    #        os.makedirs(saved_dir)

    #    file = '{}/{}_b_{}_{}.pdf'.format(saved_dir, self.args.explain,
    #                                      0, inst_id)
    #    plt.axis("off")
    #    plt_image = plt.imshow(np.transpose(expl_3c, (1, 2, 0)))
    #    plt.savefig(file, bbox_inches='tight', pad_inches=0)
    #    plt.close()
    #    print(' expl image is saved to', file)

class Args(object):
    def __init__(self, command):
        self.appr = 'lime'
        self.model = None
        self.test = None
        self.train = None
        self.complete = None
        self.tabular = False
        self.shape = None
        self.segalg = 'quickshift'
        self.nof_seg = 15
        self.visual = None
        self.verbose = 0
        self.target = None

        if command:
            self.parse(command)

        assert self.model is not None, "pls add the path to model"
        assert self.test is not None, "pls add the path to test data"

        if 'mnist' in self.test.lower():
            if self.train is None:
                self.train = self.test.replace('/test', '/train')
            if self.complete is None:
                self.complete = self.test.replace('/test', '/complete')
        else:
            if self.train is None:
                self.train = self.test.replace('/test/', '/train/').replace('_test', '_train')
            if self.complete is None:
                self.complete = self.test.replace('/test/', '/complete/').replace('_test1', '')

    def parse(self, command):
        try:
            opts, args = getopt.getopt(command[1:],
                                       'v',
                                       ['test=', 'appr=', 'model=', 'tabular',
                                        'train=', 'complete=', 'segalg=',
                                        'nof-seg=', 'visual=', 'verbose', 'target='])

        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            sys.exit(1)

        for opt, arg in opts:
            if opt == '--model':
                self.model = str(arg)
            elif opt == '--target':
                self.target = int(arg)
            elif opt in ('-v', '--verbose'):
                self.verbose += 1
            elif opt == '--segalg':
                self.segalg = str(arg)
            elif opt == '--visual':
                self.visual = str(arg)
            elif opt == '--nof-seg':
                self.nof_seg = int(arg)
            elif opt == '--test':
                self.test = str(arg)
            elif opt == '--train':
                self.train = str(arg)
            elif opt == '--complete':
                self.complete = str(arg)
            elif opt == '--appr':
                self.appr = str(arg)
            elif opt == '--tabular':
                self.tabular = True
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        self.files = args

#=============================================================================
if __name__ == "__main__":
    random.seed(1234)
    time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
           resource.getrusage(resource.RUSAGE_SELF).ru_utime

    args = Args(sys.argv)
    
    try:
        test = pd.read_csv(args.test)
    except:
        args.train = '../' + args.train
        args.test = '../' + args.test
        test = pd.read_csv(args.test)

    X_test = test.iloc[:, :-1]
    Y_test = test.iloc[:, -1]

    train = pd.read_csv(args.train)
    X_train = train.iloc[:, :-1]
    Y_train = train.iloc[:, -1]

    if '_data' in args.train:
        with open(args.train + '.pkl', 'rb') as f:
            info = pickle.load(f)
    else:
        info = None

    if 'mnist' in args.test.lower():
        if '10,10' in args.test:
            args.shape = (10, 10)
        else:
            args.shape = (28, 28)

        if 'origin' not in args.test:
            if '0.46/' in args.test:
                args.threshold = 0.46
            elif '0.43/' in args.test:
                args.threshold = 0.43
            elif '0.16/' in args.test:
                args.threshold = 0.16

            if args.shape[0] == 10:
                args.ori_test = args.test.replace(
                                '{}/'.format(args.threshold), '').replace(
                                '_data', '_origin')
            else:
                args.ori_test = args.test.replace('/test', '/test_origin')
        else:
            args.ori_test = args.test

        if 'pneumoniamnist' in args.ori_test:
            args.ori_test = args.ori_test.replace('_data.csv', '.csv')

        ori_test = pd.read_csv(args.ori_test)
        X_ori_test = ori_test.iloc[:, :-1]
        Y_ori_test = ori_test.iloc[:, -1]

        if 'origin' not in args.test:
            if args.shape[0] == 10:
                args.ori_train = args.train.replace(
                    '{}/'.format(args.threshold), '').replace(
                    '_data', '_origin')
            else:
                args.ori_train = args.test.replace('/train', '/train_origin')

            if 'pneumoniamnist' in args.ori_train:
                args.ori_train = args.ori_train.replace('_data.csv', '.csv')
        else:
            args.ori_train = args.train

        ori_train = pd.read_csv(args.ori_train)
        X_ori_train = ori_train.iloc[:, :-1]
        Y_ori_train = ori_train.iloc[:, -1]
    elif 'cifar' in args.test.lower():
        if '10,10' in args.test:
            args.shape = (10, 10)
        else:
            args.shape = (32, 32)
        args.ori_test = args.test
        ori_test = pd.read_csv(args.ori_test)
        X_ori_test = ori_test.iloc[:, :-1]
        Y_ori_test = ori_test.iloc[:, -1]
        args.ori_train = args.train
        ori_train = pd.read_csv(args.ori_train)
        X_ori_train = ori_train.iloc[:, :-1]
        Y_ori_train = ori_train.iloc[:, -1]
    else:
        X_ori_train = X_train
        X_ori_test = X_test
        Y_ori_train = Y_train
        Y_ori_test = Y_test

    options = Options(None)
    if '_data' in args.train:
        options.use_categorical = True
    options.files = [args.model]
    xgb = XGBooster(options, from_model=options.files[0])
    xgb.encode()

    hexplainer = HExplainer(args, xgb, X_train, X_ori_train, Y_train, Y_ori_train,
                            target_names=Y_train.unique(), info=info)

    if args.target is None:
        inst_ids = range(X_test.shape[0])
    else:
        inst_ids = [args.target]

    for inst_id in inst_ids:
        print('\ninst:', inst_id)
        X = X_test.iloc[inst_id:inst_id+1].to_numpy()
        X_ori = X_ori_test.iloc[inst_id:inst_id+1].to_numpy()
        hexplainer.explain(X, X_ori, inst_id)
        if inst_id >= 14:
            exit()
    exit()
