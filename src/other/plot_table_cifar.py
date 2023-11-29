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
import numpy as np
import pickle
import glob
import statistics
import json
import matplotlib.pyplot as plt
from heatmap_cifar import normalise, attr_plot, axp_stats, axp_stats2, cxp_stats, measure_dist, compare_lists
from parse_tab_logs import attr_plot as tab_attr_plot
from PIL import Image
import cv2
import pandas as pd
import statistics

#
# =============================================================================
def cp_attr_plot(gt_file, inst_id, cut2limits, update=False, flip=False, wffa=False):
    if 'pneumonia' in gt_file:
        dtname = '_pneumonia'
    elif 'cifar' in gt_file:
        dtname = '_cifar'
    else:
        dtname = '_mnist'

    if '10,10' in gt_file:
        downscale = '10,10'
    elif '14,14' in gt_file:
        downscale = '14,14'
    elif '32,32' in gt_file:
        downscale = '32,32'
    else:
        downscale ='28,28'

    shape = tuple(map(lambda l: int(l), downscale.split(',')))
    if '1,3' in gt_file:
        cls = '_1v3'
    elif '1,7' in gt_file:
        cls = '_1v7'
    elif 'ship,truck' in gt_file:
        cls = '_ship,truck'
    else:
        cls = ''

    ori = '_ori' #if 'ori' in gt_file else ''
    inst = '_inst{}'.format(inst_id)
    for cut in cut2limits:
        for limit in cut2limits[cut]:
            if limit:
                file = gt_file.replace('../stats/img/', '../stats/img/{}/{}/'.format(cut, limit))
            else:
                file = gt_file
            with open(file, 'r') as f:
                info = json.load(f)
            stats = info['stats']['inst{}'.format(inst_id)]
            saved_dir = '../plots/img/{}{}{}/'.format(downscale, dtname, cls)
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)
            new_file = '{}{}{}{}{}{}{}.pdf'.format(saved_dir, downscale, dtname, cls, inst,
                                                   '_{}{}'.format(cut, limit) if limit
                                                   else '_{}'.format('attr' if '10' in downscale else 'time7200'),
                                                   ori)
            if wffa:
                #heatmap = stats['heatmaps']['coex2']
                new_file = new_file.replace('.pdf', '_wffa.pdf')
            #else:
            #    heatmap = stats['heatmaps']['coex']
            #if update:
            #if '-3c' in file:
            ori_img = [[[stats['inst']['i_{}_{}_{}'.format(x, y, c)] for c in range(3)] for y in range(shape[0])] for x in range(shape[1])]
            ori_img = np.array(ori_img)
            #else:
            #    ori_img = np.stack(np.split(np.array(stats['inst']),
            #                                shape[0]))

            if ori_img.min() < 0:
                # step 1: convert it to [0 ,2]
                np_image = ori_img + 1
                # step 2: convert it to [0 ,1]
                np_image = np_image - np_image.min()
                ori_img = np_image / (np_image.max() - np_image.min())
            if wffa:
                lit2imprt = normalise(axp_stats2(stats['coexes']))
            else:
                lit2imprt = normalise(axp_stats(stats['coexes']))
            print('lit2imprt:', lit2imprt)
            print('wffa:' if wffa else 'ffa:')
            print('lit2imprt:', lit2imprt)
            attr_plot(ori_img, i=inst_id, shape=shape,
                     var_count=lit2imprt, newfile=new_file, prefix=new_file,
                      flip=flip, wffa=wffa)
            #else:
            #    cmd = 'cp {} {}'.format(heatmap, new_file)
            #    os.system(cmd)


def cp_attr_plot_hexp(gt_file, inst_id, update=False, flip=False, wffa=False):
    if 'pneumonia' in gt_file:
        dtname = '_pneumonia'
    elif 'cifar' in gt_file:
        dtname = '_cifar'
    else:
        dtname = '_mnist'
    if '10,10' in gt_file:
        downscale = '10,10'
    elif '14,14' in gt_file:
        downscale = '14,14'
    elif '32,32' in gt_file:
        downscale = '32,32'
    else:
        downscale = '28,28'
    shape = tuple(map(lambda l: int(l), downscale.split(',')))
    if '1,3' in gt_file:
        cls = '_1v3'
        cls_ = '-1,3'
    elif '1,7' in gt_file:
        cls = '_1v7'
        cls_ = '-1,7'
    elif 'ship,truck' in gt_file:
        cls = '_ship,truck'
        cls_ = '-ship,truck'
    else:
        cls = ''
        cls_ = ''
    ori = '_ori' if 'ori' in gt_file else ''
    inst = '_inst{}'.format(inst_id)

    with open(gt_file, 'r') as f:
        gt_info = json.load(f)

    for appr in ['lime', 'shap', 'kernelshap']:
        hexp_file = '../stats/img/bt{}-{}{}{}-t50-d3-{}.json'.format(dtname + '-10',
                                                          downscale, cls_, '-origin' if ori else '',  appr)
        hexp_file = hexp_file.replace('_', '-')
        with open(hexp_file, 'r') as f:
            info = json.load(f)

        stats = info['stats']['inst{}'.format(inst_id)]
        #heatmap = stats['heatmap']
        saved_dir = '../plots/img/{}{}{}/'.format(downscale, dtname, cls)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        new_file = '{}{}{}{}{}{}{}.pdf'.format(saved_dir, downscale, dtname, cls, inst, '_{}'.format(appr),
                                                    ori)
        #if update:
        ori_img = [[[gt_info['stats'][inst.replace('_', '')]['inst']['i_{}_{}_{}'.format(x, y, c)] for c in range(3)] for y in range(shape[0])] for x in
                   range(shape[1])]
        ori_img = np.array(ori_img)
        if ori_img.min() < 0:
            # step 1: convert it to [0 ,2]
            np_image = ori_img + 1
            # step 2: convert it to [0 ,1]
            np_image = np_image - np_image.min()
            ori_img = np_image / (np_image.max() - np_image.min())
        lit2imprt = {abs(int(lit)): abs(imprt) for lit, imprt in stats['lit2imprt'].items()}
        nor_lit2imprt = normalise(lit2imprt)
        attr_plot(ori_img, i=inst_id, shape=shape,
                  var_count=nor_lit2imprt, newfile=new_file, prefix=new_file,
                  flip=flip)
        #else:
        #    cmd = 'cp {} {}'.format(heatmap, new_file)
        #    os.system(cmd)

def cp_attr_plot_tabular(dtname, file, inst_id, update=False, wffa=False):
    with open(file, 'r') as f:
        info = json.load(f)
    stats = info['stats']['inst{}'.format(inst_id)]
    #heatmap = stats['heatmaps']["nor-attr"]
    saved_dir = '../plots/tab/'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    new_file = '{}formal_{}_inst{}.pdf'.format(saved_dir, dtname, inst_id)
    if wffa:
        new_file = new_file.replace('.pdf', '_wffa.pdf')

    #if update:
    features = stats['inst']
    if wffa:
        fid2imprt = normalise(axp_stats2(stats['coexes']))
    else:
        fid2imprt = normalise(axp_stats(stats['coexes']))

    fid2imprt = {abs(int(fid)): abs(imprt) for fid, imprt in fid2imprt.items()}
    for fid in range(len(features)):
        if fid not in fid2imprt:
            fid2imprt[fid] = 0.00
    names = []
    values = []
    for fid in sorted(fid2imprt.keys(), key=lambda l: (fid2imprt[l], l)):
        names.append(features[fid].replace(' == ', ': '))
        values.append(fid2imprt[fid])
    tab_attr_plot(features, fid2imprt, prefix='', suffix='',
                  newfile=new_file, names=names, values=values)
    #else:
    #cmd = 'cp {} {}'.format(heatmap, new_file)
    #os.system(cmd)

def cp_attr_plot_tabular_hexp(dtname, gt_file, inst_id, update=False, wffa=False):
    for appr in ['lime', 'shap', 'kernelshap']:
        file = '../stats/tab/bt-{}-{}.json'.format(dtname, appr)
        with open(file, 'r') as f:
            info = json.load(f)
        stats = info['stats']['inst{}'.format(inst_id)]
        #heatmap = stats['heatmaps']["nor-attr"]
        saved_dir = '../plots/tab/'
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        new_file = '{}{}_{}_inst{}.pdf'.format(saved_dir, appr, dtname, inst_id)
        if wffa:
            new_file = new_file.replace('.pdf', '_wffa.pdf')

        #if update:
        with open(gt_file, 'r') as f:
            gt_info = json.load(f)
        features = gt_info['stats']['inst{}'.format(inst_id)]['inst']
        if wffa:
            gt_fid2imprt = normalise(axp_stats2(gt_info['stats']['inst{}'.format(inst_id)]['coexes']))
        else:
            gt_fid2imprt = normalise(axp_stats(gt_info['stats']['inst{}'.format(inst_id)]['coexes']))
        gt_fid2imprt = {abs(int(fid)): abs(imprt) for fid, imprt in gt_fid2imprt.items()}

        nor_fid2imprt = {abs(int(fid)): abs(imprt) for fid, imprt in stats['nor-lit2imprt'].items()}
        for fid in range(len(features)):
            if fid not in gt_fid2imprt:
                gt_fid2imprt[fid] = 0.00
            if fid not in nor_fid2imprt:
                nor_fid2imprt[fid] = 0.00

        names = []
        values = []
        for fid in sorted(nor_fid2imprt.keys(), key=lambda l: (gt_fid2imprt.get(l, 0), l)):
            names.append(features[fid].replace(' == ', ': '))
            values.append(nor_fid2imprt[fid])
        tab_attr_plot(features, nor_fid2imprt, prefix='', suffix='',
                      newfile=new_file, names=names, values=values)
        #else:
        #    cmd = 'cp {} {}'.format(heatmap, new_file)
        #    os.system(cmd)

def csv2latex_rank(files, cutoffs, i):
    new_columns = ['LIME', 'SHAP'] + cutoffs
    if i == '2':
        new_columns_ = ['LIME', 'SHAP'] + ['\\wffalabel{' + str(c) + '}' for c in cutoffs]
    else:
        new_columns_ = ['LIME', 'SHAP'] + ['\\ffalabel{' + str(c) + '}' for c in cutoffs]
    res = collections.defaultdict(dict)
    metrics = set()
    for file_info in files:
        cls = file_info['cls']
        file = file_info['file']
        dtname = file_info['dtname']
        metric = file_info['metric']
        downscale = file_info['downscale']
        shape = list(map(lambda l: int(l), downscale.split(',')))
        df = pd.read_csv(file)
        #print(df[df.columns[0]])
        #print(type(df[df.columns[0]]))
        if 'error' in metric[0]:
            #avg = {col: statistics.mean(df[col.lower()].to_numpy()[:15]) / (shape[0] * shape[1]) for col in new_columns}
            avg = {col: statistics.mean(df[col.lower()].to_numpy()[:15]) for col in new_columns}
        else:
            avg = {col: statistics.mean(df[col.lower()].to_numpy()[:15]) for col in new_columns}
        res['{}-{}{}'.format(downscale, dtname, '-' + cls if cls else '')][metric[1]] = avg
        metrics.add(metric[1])

    def func(s):
        if 'rbo' in s:
            return 3
        elif 'tau' in s:
            return 2
        else:
            return 1
    metrics = sorted(metrics, key=lambda l: func(l.lower()))
    #metrics_ = list(map(lambda l: 'tau' if 'tau' in l else l, metrics))

    cap = 'Comparison in {} * {} Images'.format(downscale.split(',')[0].strip(), downscale.split(',')[1].strip())

    rows = ['\\begin{table}[t!]',
            '\\caption{' + cap + '}',
            '\\centering',
            '%\\begin{adjustbox}{center}',
            '%\\label{tab:inst}',
            '%\\scalebox{0.83}{',
            '	%\\setlength{\\tabcolsep}{1.9pt}',
            '	\\begin{tabular}{' + 'c'*(len(new_columns) + 1) + '}  \\toprule']
    # add header
    rows.append(' & '.join(['\\textbf{Dataset}'] + ['\\textbf{' + col + '}' for col in new_columns_]) + '  \\\\ \\midrule')
    for i, metric in enumerate(metrics):
        rows.append(' & \\multicolumn{' + '{}'.format(len(new_columns)) + '}{c}{\\textbf{' + metric + '}}  \\\\ \\midrule')
        rows[-1] = rows[-1].replace('manhattan', 'Error').replace('kendalltau', 'Kendall’s Tau').replace('rbo', 'RBO')

        for dt in res:
            row = [dt] + ['{:.2f}'.format(res[dt][metric][col]) for col in new_columns]
            row = ' & '.join(row) + '  \\\\'
            rows.append(row)
        if i != len(metrics)-1:
            rows[-1] += '  \\midrule'
        else:
            rows[-1] += '  \\bottomrule'
    end = ['	\\end{tabular}',
            '%}',
            '%\\end{adjustbox}',
            '\\end{table}']
    rows.extend(end)
    #print('\n'. join(rows))
    return '\n'. join(rows)

def csv2latex_tab(files):
    cap = 'Ranking comparison in tabular data'
    rows = ['\\begin{table}[t!]',
            '\\caption{' + cap + '}',
            '\\centering',
            '%\\begin{adjustbox}{center}',
            '%\\label{tab:inst}',
            '\\scalebox{0.8}{',
            '	\\setlength{\\tabcolsep}{1.2pt}']

    for ii, file_info in enumerate(files):
        file = file_info['file']
        metric = file_info['metric']

        df = pd.read_csv(file)
        dtnames = df.columns[1:]
        insts = []
        for dtname in dtnames:
            with open('../stats/tab/bt-{}-formal-con.json'.format(dtname), 'r') as f:
                info = json.load(f)
            insts.append(info['stats']['inst0']['inst'])
        if ii == 0:
            rows.append('	\\begin{tabular}{' + 'c' * (len(df.columns)) + '}  \\toprule')

            rows.append(' & '.join(['\\textbf{Dataset}'] +
                             ['\\textbf{' + col  + '}' for col in df.columns.to_list()[1:]]) + \
                  '  \\\\ ')
            rows.append(' & '.join([' $|\\fml{F}|$ '] + ['(' + str(len(inst)) + ')'
                                            for inst in insts]) + '  \\\\ \\midrule')
            #rows.append('\\textbf{Approach} & \\multicolumn{' + '{}'.format(
            #            len(df.columns) - 1) + '}{c}{\\textbf{' + metric[1] + '}} \\\\ \\midrule')
            metric_ = 'Error'
        else:
            if 'tau' in metric[1].lower():
                metric_ = 'Kendall’s Tau'
            else:
                metric_ = metric[1].upper()

        if ii == 0:
            rows.append(' \\textbf{Approach} & \\multicolumn{' + '{}'.format(
                len(df.columns) - 1) + '}{c}{\\textbf{' + metric_ + '}} \\\\ \\midrule')
        else:
            rows.append(' & \\multicolumn{' + '{}'.format(
                len(df.columns) - 1) + '}{c}{\\textbf{' + metric_ + '}} \\\\ \\midrule')

        for i in range(df.shape[0]):
            row = df.iloc[i, :].to_list()
            row[0] = row[0].upper()
            row[1:] = ['{:.2f}'.format(value if ii != 0 else value ) for value in row[1:]]
            rows.append(' & '.join(row) + '  \\\\')

        if ii != len(files)-1:
            rows[-1] += '  \\midrule'
        else:
            rows[-1] += '  \\bottomrule'

    end = ['	\\end{tabular}',
       '}',
       '%\\end{adjustbox}',
       '\\end{table}']
    rows.extend(end)
    #print('\n'. join(rows))
    return '\n'. join(rows)

def csv2latex_jit(files, metrics):
    res = collections.defaultdict(lambda : collections.defaultdict(dict))
    datasets = set()
    for file_info in files:
        file = file_info['file']
        metric = file_info['metric']
        df = pd.read_csv(file)
        datasets = datasets.union(df.columns[1:])
        for i in range(df.shape[0]):
            appr = df.iloc[i, 0].upper()
            for j in range(1, df.shape[1]):
                res[appr][df.columns[j]][metric] = df.iloc[i, j]
    datasets = sorted(datasets)
    insts = []
    for dtname in datasets:
        with open('../stats/jit/{}-LR-formal.json'.format(dtname), 'r') as f:
            info = json.load(f)
        insts.append(info['stats']['inst0']['inst'])

    apprs = sorted(res.keys(), key=lambda l: 'SHAP' in l)

    cap = 'Just in time comparison in LIME and SHAP'
    rows = ['\\begin{table}[t!]',
            '\\caption{' + cap + '}',
            '%\\label{tab:jit}',
            '\\centering',
            '%\\begin{adjustbox}{center}',
            '%\\scalebox{0.83}{',
            '	%\\setlength{\\tabcolsep}{3pt}',
            '	\\begin{tabular}{' + 'c' * (len(metrics) * 2 + 1) + '}  \\toprule']
    rows.append(' & '.join(['\\multirow{2}{*}{\\textbf{Approach}}'] +
                            ['\\multicolumn{' + '{}'.format(len(metrics)) + '}{c}{\\textbf{' + dt + '}~\\small($|\\fml{F}|$ ' +
                             str(len(inst)) + ')}' for dt, inst in zip(datasets, insts)]) +
                '  \\\\ \\cmidrule{2-' + '{}'.format(1 + len(metrics) * 2) + '}')

    row = ' & '.join([' '] + ['\\textbf{Error}' if 'error' in metric[0]
                              else '\\textbf{' + metric[1] + '}'
                              for metric in metrics] * 2) + '  \\\\ \\midrule'
    rows.append(row)
    for appr in apprs:
        row = [appr]
        for dt in datasets:
            for metric in metrics:
                val = res[appr][dt][metric]
                row.append('{:.2f}'.format(val if 'error' not in metric[0]
                                           else val * 1))
        rows.append(' & '.join(row) + '  \\\\')
    rows[-1] += '  \\bottomrule'

    end = ['	\\end{tabular}',
       '%}',
       '%\\end{adjustbox}',
       '\\end{table}']
    rows.extend(end)
    #print('\n'. join(rows))
    return '\n'. join(rows)

def avg_stats(file, nof_insts=15):
    with open(file, 'r') as f:
        info = json.load(f)
    rtimes = []
    coexes = []
    for inst in info['stats']:
        if '10,10' in file:
            if info['stats'][inst]['status2']:
                rtimes.append(info['stats'][inst]['rtime'])
                coexes.append(len(info['stats'][inst]['coexes']))
                if len(rtimes) >= nof_insts:
                    break
        else:
            rtimes.append(info['stats'][inst]['rtime'])
            coexes.append(len(info['stats'][inst]['coexes']))
            if len(rtimes) >= nof_insts:
                break
    rtimes.sort()
    print('rtimes:', rtimes)
    print('avg rtime:', statistics.mean(rtimes))
    coexes.sort()
    print('coexes[0]:', coexes[0])
    print('coexes[-1]:', coexes[-1])
    print('avg coexes:', statistics.mean(coexes))
    return statistics.mean(rtimes), statistics.mean(coexes)


#
# =============================================================================
if __name__ == "__main__":
    """
    Produce image feature attribution plot
    """

    file2inst = {'../stats/img/bt-cifar-10-32,32-ship,truck-formal-t50-d3-con.json': [6]}

    cut2limits = {'time': [30, 60, 120, 300, 600, 1200, 1800, 3600, None]}

    for file, inst_ids in file2inst.items():
        #print(file)
        #flip = 'pneu' not in file
        flip = False
        for wffa in [True, False]: #, True]:
            for inst_id in inst_ids:
                cp_attr_plot(file, inst_id, cut2limits, update=True, flip=flip, wffa=wffa)
                cp_attr_plot_hexp(file, inst_id, update=True, flip=flip)#, wffa=wffa)
    """
       compas example
   """
    for dtname in ['compas']:
        formal_file = '../stats/tab/bt-{}-formal-con.json'.format(dtname)
        wffa = True
        #cp_attr_plot_tabular(dtname, formal_file, inst_id=1, update=True, wffa=wffa)  # False)
        cp_attr_plot_tabular_hexp(dtname, formal_file, inst_id=1, update=True, wffa=wffa)