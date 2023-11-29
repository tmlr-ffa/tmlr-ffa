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
from heatmap import normalise, attr_plot, axp_stats, axp_stats2, cxp_stats, measure_dist, compare_lists
from parse_tab_logs import attr_plot as tab_attr_plot
from PIL import Image
import cv2
import pandas as pd
import statistics

#
# =============================================================================
def cp_attr_plot(gt_file, inst_id, cut2limits, update=False, flip=False, wffa=False):
    dtname = '_pneumonia' if 'pneumonia' in gt_file else '_mnist'
    downscale = '10,10' if '10,10' in gt_file else '28,28'
    shape = tuple(map(lambda l: int(l), downscale.split(',')))
    if '1,3' in gt_file:
        cls = '_1v3'
    elif '1,7' in gt_file:
        cls = '_1v7'
    else:
        cls = ''
    ori = '_ori' if 'ori' in gt_file else ''
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
            ori_img = np.stack(np.split(np.array(stats['inst']),
                                        shape[0]))
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

            attr_plot(ori_img, i=inst_id, shape=shape,
                     var_count=lit2imprt, newfile=new_file, prefix=new_file,
                      flip=flip, wffa=wffa)
            #else:
            #    cmd = 'cp {} {}'.format(heatmap, new_file)
            #    os.system(cmd)


def cp_attr_plot_hexp(gt_file, inst_id, update=False, flip=False, wffa=False):
    dtname = '_pneumonia' if 'pneumonia' in gt_file else '_mnist'
    downscale = '10,10' if '10,10' in gt_file else '28,28'
    shape = tuple(map(lambda l: int(l), downscale.split(',')))
    if '1,3' in gt_file:
        cls = '_1v3'
        cls_ = '-1,3'
    elif '1,7' in gt_file:
        cls = '_1v7'
        cls_ = '-1,7'
    else:
        cls = ''
        cls_ = ''
    ori = '_ori' if 'ori' in gt_file else ''
    inst = '_inst{}'.format(inst_id)

    #with open(gt_file, 'r') as f:
    #    gt_info = json.load(f)

    for appr in ['fastshap', 'kernelshap']: #'lime', 'shap', ]:
        hexp_file = '../stats/img/bt{}-{}{}{}-{}.json'.format(dtname if dtname == '_mnist' else dtname + 'mnist',
                                                          downscale, cls_, '-origin' if ori else '',  appr)
        hexp_file = hexp_file.replace('_', '-')
        with open(hexp_file, 'r') as f:
            info = json.load(f)

        saved_dir = '../plots/img/{}{}{}/'.format(downscale, dtname, cls)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        new_file = '{}{}{}{}{}{}{}.pdf'.format(saved_dir, downscale, dtname, cls, inst, '_{}'.format(appr),
                                                    ori)
        try:
            stats = info['stats']['inst{}'.format(inst_id)]
            #heatmap = stats['heatmap']
            #if update:
            ori_img = np.array(info['stats']['inst{}'.format(inst_id)]['inst'])
            ori_img = np.stack(np.split(ori_img, shape[0]))
        except:
            try:
                df = pd.read_csv('../datasets/mnist/10,10/1,7/test_origin_data.csv')
                ori_img = df.iloc[inst_id, :-1]
                ori_img = np.stack(np.split(ori_img, shape[0]))
            except:
                return

        if ori_img.min() < 0:
            # step 1: convert it to [0 ,2]
            np_image = ori_img + 1
            # step 2: convert it to [0 ,1]
            np_image = np_image - np_image.min()
            ori_img = np_image / (np_image.max() - np_image.min())

        lit2imprt = {abs(int(lit)): abs(imprt) for lit, imprt in stats['lit2imprt2'].items()}
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
    for appr in ['kernelshap', 'fastshap', 'shapreg']: #['lime', 'shap']:
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
    new_columns = ['LIME', 'SHAP', 'KERNELSHAP'] + cutoffs
    if i == '2':
        new_columns_ = ['LIME', 'SHAP', 'KERNELSHAP'] + ['\\wffalabel{' + str(c) + '}' for c in cutoffs]
    else:
        new_columns_ = ['LIME', 'SHAP', 'KERNELSHAP'] + ['\\ffalabel{' + str(c) + '}' for c in cutoffs]
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
    image error and rank
    """
    latex = []
    for i in ['', '2']:
        metrics = [('errors' + i, 'manhattan'),
                   ('coefs' + i, 'kendalltau'),
                   ('coefs' + i, 'rbo')]

        for downscale in ['32,32']:#10,10', '28,28']:
            cutoffs = ['30', '60', '120', '600', '1200', '3600']
            #if '28' in downscale:
            #    cutoffs = ['10', '30', '120', '600', '1200', '3600']
            #else:
            #    cutoffs = ['10', '30', '60', '120', '600', '1200']
            files = []
            for metric in metrics:
                for dtname in ['cifar-10']:#mnist','pneumoniamnist']:
                    classes = ['ship,truck']#1,3', '1,7'] if dtname == 'mnist' else ['']
                    for cls in classes:
                        file = '../stats/tables/img/{}-{}{}-{}-{}.csv'.format(dtname, downscale, '-' + cls if cls else '',
                                                                            metric[0], metric[1])
                        files.append({'file': file,
                                      'dtname': dtname,
                                      'cls': cls,
                                      'metric': metric,
                                      'downscale': downscale})
            latex.append(csv2latex_rank(files, cutoffs, i))
    saved_dir = '../stats/latex/'
    if not os.path.isdir(saved_dir):
       os.makedirs(saved_dir)
    with open(saved_dir + 'cifar-10-img-cmpr.txt', 'w') as f:
       f.write('\n\n'.join(latex))

    exit()

    #"""
    #runtime
    #"""
    #files = ['../stats/img/bt-mnist-28,28-1,3-origin-formal-con.json',
    # '../stats/img/bt-mnist-28,28-1,7-origin-formal-con.json',
    # '../stats/img/bt-pneumoniamnist-28,28-origin-formal-con.json',
    # '../stats/img/bt-mnist-10,10-1,3-origin-formal-con.json',
    # '../stats/img/bt-mnist-10,10-1,7-origin-formal-con.json',
    # '../stats/img/bt-pneumoniamnist-10,10-origin-formal-con.json']

    #files_ = glob.glob('../stats/tab/*formal*.json')
    #files_ = sorted(filter(lambda l: 'mnist' not in l,
    #                       files_))
    #for file in files:
    #    print(file)
    #    avg_stats(file, nof_insts=15)
    #    pass
    #exit()

    #avg_rtimes = []
    #avg_axpss = []
    #for file in files_:
    #    print(file)
    #    avg_rtime, avg_axps = avg_stats(file)
    #    avg_rtimes.append(avg_rtime)
    #    avg_axpss.append(avg_axps)
    #avg_rtimes.sort()
    #avg_axpss.sort()
    #print(avg_rtimes)
    #print(statistics.mean(avg_rtimes))
    #print()
    #print(avg_axpss)
    #print(statistics.mean(avg_axpss))
    #print()
    #exit()


    exit()