#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

# imported modules:
# ==============================================================================
from __future__ import print_function
import collections
import resource
import sys
import os
import pickle
import glob
import statistics
import json
import matplotlib.pyplot as plt
from heatmap import attr_plot, normalise, axp_stats, axp_stats2, measure_dist, compare_lists


#
# =============================================================================
def parse_tab_formal_log(log):
    model = 'lr'
    dtname = log.rsplit('/', maxsplit=1)[-1].split('_')[0]
    #bg = True if '_bg' in log else False
    #smallest = True if '_min' in log else False
    #xtype = 'abd' if '_con' not in log else 'con'

    #conf = 'formal-{}{}'.format(model + '-' if model != '' else '',
    #                                  dtname)
    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-')
    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    lines = open(log, 'r').readlines()

    for i, line in enumerate(lines):
        if 'Explaining:' in line:
            lines = lines[i:]
            break
    else:
        print('something wrong')
        exit()

    insts = []
    for i, line in enumerate(lines):
        if 'Explaining:' in line:
            insts.append(i)
    insts.append(len(lines))

    inst_feats = lines[:]
    inst_feats = list(filter(lambda l: 'Explaining:' in l, inst_feats))
    inst_feats = [l.replace('"', '').replace('IF ', '').split(':', maxsplit=1)[-1].split(' THEN ')[0].strip('" ').strip().split(' AND ') for l in inst_feats]

    for i in range(len(insts) - 1):
        inst = 'inst{0}'.format(i)
        stats = {}
        info['stats'][inst] = stats
        stats['inst'] = inst_feats[i]
        stats['status'] = True

        feat2id = {feat: fid for fid, feat in enumerate(inst_feats[i])}
        axps = []
        time = 0
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'abd:' in line:
                axp = line.split(':', maxsplit=1)[-1].replace('IF ', '').split(' THEN ')[0].strip('" ').strip().split(' AND ')
                axp = sorted(map(lambda l: feat2id[l.strip()], axp))
                axps.append(axp)
            elif 'time:' in line:
                time += float(line.split(':', maxsplit=1)[-1])
        stats['rtime'] = time
        stats['axps'] = axps

    saved_dir = '../stats/jit'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    with open(saved_dir + '/' + conf + '.json', 'w') as f:
        json.dump(info, f, indent=4)

def parse_tab_hexp_log(log):
    #dtname, model, appr = log.rsplit('/', maxsplit=1)[-1].split('_')
    #model = model.lower()
    #appr = appr.rsplit('.', maxsplit=1)[0]
    #assert appr in ('lime', 'shap')
    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-')
    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    with open(log, 'r') as f:
        lines = f.readlines()

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    for i in range(len(insts) - 1):
        inst_id = int(lines[insts[i]].split(':')[-1])
        inst = 'inst{}'.format(inst_id)
        stats = {'status': True}
        info['stats'][inst] = stats
        for ii in range(insts[i], insts[i+1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_f = line.split(':', maxsplit=1)[-1].strip().split( 'AND' )
                stats['inst'] = list(map(lambda l: l.strip(), inst_f))
            elif 'lit2imprt:' in line:
                lit2imprt = line.split(':', maxsplit=1)[-1].strip().strip('{}').split(',')
                lit2imprt = map(lambda l: l.split(':'), lit2imprt)
                lit2imprt = {int(l[0]): float(l[1]) for l in lit2imprt}
                stats['lit2imprt'] = lit2imprt
                cnt_hexp = {abs(int(lit)): abs(imprt) for lit, imprt in stats['lit2imprt'].items()}
                cnt_hexp_nor = normalise(cnt_hexp)
                stats['nor-lit2imprt'] = cnt_hexp_nor #normalise(lit2imprt, min_v=0)
                stats['pos-lit2imprt'] = {lit: imprt for lit, imprt in stats['lit2imprt'].items() if imprt > 0}
                stats['nor-pos-lit2imprt'] = normalise(stats['pos-lit2imprt'], min_v=0)
            elif ' time:' in line:
                stats['rtime'] = float(line.split(':', maxsplit=1)[-1])

    saved_dir = '../stats/jit'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{}/{}.json'.format(saved_dir, label)
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

def calculate_metrics_formal(file):
    with open(file, 'r') as f:
        info = json.load(f)
    dtname = file.rsplit('/', maxsplit=1)[-1].split('-')[0]

    for inst in info['stats']:
        #print(inst)
        inst_id = int(inst[4:])
        stats = info['stats'][inst]
        features = stats['inst']
        axps = stats['axps']
        stats['fid2imprt'] = axp_stats(axps)
        stats['nor-fid2imprt'] = normalise(stats['fid2imprt'])
        stats['fid2imprt2'] = axp_stats2(axps)
        stats['nor-fid2imprt2'] = normalise(stats['fid2imprt2'])
        #saved_dir = '../plots/lr/'
        #if not os.path.isdir(saved_dir):
        #    os.makedirs(saved_dir)
        #prefix = '{}{}_{}'.format(saved_dir, dtname, inst_id)
        #heatmaps = {}
        #heatmaps['attr'] = attr_plot(features, stats['fid2imprt'], prefix=prefix, suffix='_attr')
        #heatmaps['nor-attr'] = attr_plot(features, stats['nor-fid2imprt'], prefix=prefix, suffix='_nor_attr')
        #heatmaps['attr2'] = attr_plot(features, stats['fid2imprt2'], prefix=prefix, suffix='_attr2')
        #heatmaps['nor-attr2'] = attr_plot(features, stats['nor-fid2imprt2'], prefix=prefix, suffix='_nor_attr2')
        #stats['heatmaps'] = heatmaps

    with open(file, 'w') as f:
        json.dump(info, f, indent=4)

def calculate_metrics_hexp(file):
    with open(file, 'r') as f:
        info = json.load(f)

    appr = file.rsplit('-', maxsplit=1)[-1].split('.', maxsplit=1)[0]
    #ground truth
    gt_file = file.replace(appr, 'formal')
    with open(gt_file, 'r') as f:
        gt_info = json.load(f)

    for inst in gt_info['stats']:
        #print(inst)
        inst_id = int(inst[4:])
        stats = info['stats'][inst]
        gt_stats = gt_info['stats'][inst]
        features = gt_stats['inst']

        #saved_dir = '../plots/lr/hexp/'
        #if not os.path.isdir(saved_dir):
        #    os.makedirs(saved_dir)
        #prefix = '{}{}_{}_{}'.format(saved_dir, appr, dtname, inst_id)

        #heatmaps = {}
        #heatmaps['attr'] = attr_plot(features, stats['lit2imprt'], prefix=prefix, suffix='_attr')
        #cnt_hexp = {abs(int(lit)): abs(imprt) for lit, imprt in stats['lit2imprt'].items()}
        #cnt_hexp_nor = normalise(cnt_hexp)
        #heatmaps['nor-attr'] = attr_plot(features, cnt_hexp_nor, prefix=prefix, suffix='_nor_attr')
        #stats['heatmaps'] = heatmaps

        # compute errors, tau, rbo
        errors = {}
        errors2 = {}
        avg_errors = {}
        avg_errors2 = {}
        coefs = {}
        coefs2 = {}

        shape = (len(features), 1)
        cnt_gt = {abs(int(lit)): abs(imprt) for lit, imprt in gt_stats['nor-fid2imprt'].items()}
        cnt_gt2 = {abs(int(lit)): abs(imprt) for lit, imprt in gt_stats['nor-fid2imprt2'].items()}
        cnt_hexp = {abs(int(lit)): abs(imprt) for lit, imprt in stats['lit2imprt'].items()}
        cnt_hexp_nor = normalise(cnt_hexp)

        for metric in ['manhattan', 'euclidean']:
            errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric)
            avg_errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric, avg=True)
            errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric)
            avg_errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric, avg=True)

        for metric in ('kendall_tau', 'rbo'):
            #if metric == 'rbo':
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.99))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.9))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.8))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.7))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.5))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.1))
            #    print()
            #    print(compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.99))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.9))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.8))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.7))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.5))
            #    print(compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.1))
            #    exit()
            coef = compare_lists(cnt_hexp_nor, cnt_gt, metric=metric, p=0.75)
            coefs[metric.replace('_', '')] = coef
            coef2 = compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric, p=0.75)
            coefs2[metric.replace('_', '')] = coef2

        stats['errors'] = errors
        stats['avg-errors'] = avg_errors
        stats['errors2'] = errors2
        stats['avg-errors2'] = avg_errors2
        stats['coefs'] = coefs
        stats['coefs2'] = coefs2

    with open(file, 'w') as f:
        json.dump(info, f, indent=4)

#
# =============================================================================
if __name__ == "__main__":
    """
    parse formal tabular data logs
    """
    logs = sorted(glob.glob('../logs/jit/*formal*.log'))
    for log in logs:
        print(log)
        parse_tab_formal_log(log)

    files = sorted(glob.glob('../stats/jit/*formal*.json'))
    for file in files:
        print(file)
        calculate_metrics_formal(file)

    """
    parse tabular data hexp logs
    """
    logs = sorted(glob.glob('../logs/jit/*.log'))
    logs = sorted(filter(lambda l: 'lime' in l or 'shap' in l, logs))
    for log in logs:
        print(log)
        parse_tab_hexp_log(log)

    files = glob.glob('../stats/jit/*.json')
    files = sorted(filter(lambda l: 'shap' in l or 'lime' in l, files))
    for file in files:
        print(file)
        calculate_metrics_hexp(file)
    exit()