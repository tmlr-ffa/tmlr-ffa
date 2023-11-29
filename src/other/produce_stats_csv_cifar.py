#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

# imported modules:
# ==============================================================================
from __future__ import print_function
import pandas as pd
import glob
import os
import collections
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics


def extract_stats(gt_file, cut2limits):
    limit2stats = {}
    for cut, limits in cut2limits.items():
        for limit in limits:
            file = gt_file.replace('stats/img/', '/stats/img/{}/{}/'.format(cut, limit))
            with open(file, 'r') as f:
                info = json.load(f)
            stats = info['stats']

            if '28,28' in gt_file or '32,32' in gt_file:
                insts = sorted(info['stats'].keys(), key=lambda l: int(l[4:]))
            else:
                insts = filter(lambda l: info['stats'][l]['status2'], info['stats'].keys())
                insts = sorted(insts, key=lambda l: int(l[4:]))

            errors = collections.defaultdict(dict)
            coefs = collections.defaultdict(dict)
            for inst in insts:
                inst_id = int(inst[4:])
                errors['errors'][inst_id] = stats[inst]['errors']['coexes']
                errors['errors2'][inst_id] = stats[inst]['errors2']['coexes']
                coefs['coefs'][inst_id] = stats[inst]['coefs']['coexes']
                coefs['coefs2'][inst_id] = stats[inst]['coefs2']['coexes']
            limit2stats[limit] = {'coefs': coefs,
                                  'errors': errors}
    return limit2stats

def extract_stats_hexp(hexp_file):
    print(hexp_file)
    with open(hexp_file, 'r') as f:
        info = json.load(f)
    stats = info['stats']
    h_stats = {}
    for inst in stats:
        inst_id = int(inst[4:])
        try:
            errors = stats[inst]['errors']
        except:
            continue
        errors2 = stats[inst]['errors2']
        coefs = stats[inst]['coefs']
        coefs2 = stats[inst]['coefs2']
        h_stats[inst_id] = {'errors': errors, 'errors2': errors2,
                          'coefs': coefs, 'coefs2': coefs2}
    return h_stats

def gnrt_table(filename, formal_stats, hexp_stats, metric):
    iserror = 'error' in metric[0]
    res_dict = collections.defaultdict(list)
    columns = []
    m_type = metric[0][:-1] if metric[0].endswith('2') else metric[0]
    m_name = metric[1]
    insts = None
    # formal approach
    for limit in formal_stats:
        formal_s = formal_stats[limit][m_type][metric[0]]
        if 'inst' not in res_dict:
            res_dict['inst'] = list(map(lambda l: 'inst{}'.format(l), sorted(formal_s.keys())))
        if insts is None:
            insts = sorted(formal_s.keys())
        for inst_id in sorted(formal_s.keys()):
            res = formal_s[inst_id][m_name]
            res_dict[limit].append(res)

    # lime and shap
    for appr in hexp_stats:
        h_stats = hexp_stats[appr]
        for inst_id in insts:
            res = h_stats[inst_id][metric[0]][metric[1]]
            res_dict[appr].append(res)

    columns = ['inst'] + list(hexp_stats.keys()) + sorted(formal_stats.keys())

    data = {}
    data[columns[0]] = res_dict[columns[0]] + ['average']
    for col in columns[1:]:
        data[col] = res_dict[col] + [sum(res_dict[col]) / len(res_dict[col])]
        data[col] = list(map(lambda l: '{:.2f}'.format(l) if iserror else '{:.4f}'.format(l), data[col]))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(filename.replace('.pdf', '.csv'), index=False)

    ## print(df)
    #fig, ax = plt.subplots(figsize=(12, 4))
    #ax.axis('tight')
    #ax.axis('off')
    #the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    #pp = PdfPages(filename)
    #pp.savefig(fig, bbox_inches='tight')
    #pp.close()
    #plt.close()

def extract_tab_stats(files):
    h_stats = collections.defaultdict(lambda : collections.defaultdict(dict))
    for file in files:
        if '/jit/' not in file:
            appr = file.rsplit('-', maxsplit=1)[-1].split('.', maxsplit=1)[0]
            dtnanme = file.rsplit('/', maxsplit=1)[-1].split('-', maxsplit=1)[-1]
            dtname = dtnanme.split('-'+appr, maxsplit=1)[0]
        else:
            appr = file.rsplit('-', maxsplit=1)[-1].split('.', maxsplit=1)[0]
            dtname = file.rsplit('/', maxsplit=1)[-1].split('-', maxsplit=1)[0]

        with open(file, 'r') as f:
            info = json.load(f)
        for inst in info['stats']:
            stats = info['stats'][inst]
            inst_id = int(inst[4:])
            try:
                errors = stats['errors']
            except:
                continue
            errors2 = stats['errors2']
            coefs = stats['coefs']
            coefs2 = stats['coefs2']
            nof_feats = len(info['stats'][inst]['inst'])
            h_stats[appr][dtname][inst_id] = {'errors': errors, 'errors2': errors2,
                                              'coefs': coefs, 'coefs2': coefs2,
                                              'nof-feats': nof_feats}
    return h_stats


def gnrt_tab_table(filename, stats, metric):
    for appr in stats:
        columns = [' '] + sorted(stats[appr].keys())
        break
    rows = []
    for appr in stats:
        row = [appr]
        for dt in columns[1:]:
            # print(stats[appr][dt])
            # if 'error' in metric[0]:
            #    res = [stats[appr][dt][inst_id][metric[0]][metric[1]] / stats[appr][dt][inst_id]['nof-feats']
            #           for inst_id in stats[appr][dt]]
            # else:
            res = [stats[appr][dt][inst_id][metric[0]][metric[1]]
                   for inst_id in stats[appr][dt]]
            res = statistics.mean(res)
            row.append('{}'.format(res) if 'error' in metric[0] else '{:.4f}'.format(res))

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filename.replace('.pdf', '.csv'), index=False)

    #fig, ax = plt.subplots(figsize=(12, 2))
    #ax.axis('off')
    #ax.axis('tight')
    #the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    #pp = PdfPages(filename)
    #pp.savefig(fig, bbox_inches='tight')
    #pp.close()
    #plt.close()

def gnrt_img_metrics():
    """
    Generate image metrics
    """
    cut2limits = {'time': [30, 60, 120, 300, 600, 1200, 1800, 3600]}
    gt_origin_files = sorted(glob.glob('../stats/img/*cifar*t50*d3*.json'))
    appr2gt_ori_files = collections.defaultdict(list)
    for file in gt_origin_files:
        if 'lime' in file:
            appr = 'lime'
        elif 'kernelshap' in file:
            appr = 'kernelshap'
        elif 'shap' in file:
            appr = 'shap'
        else:
            appr = 'formal'
        appr2gt_ori_files[appr].append(file)

    apprs = sorted(set(appr2gt_ori_files.keys()).difference(['formal']))

    """
    Extract stats
    """
    formals_ori = []
    hexps_ori = []
    for formal_file in appr2gt_ori_files['formal']:
        #if '10,10' in formal_file:
        #    shape = (10, 10)
        #else:
        #    shape = (28, 28)
        hexp_ori_stats = {}
        for appr in apprs:#['lime', 'shap']:
            h_file = formal_file.replace('-formal', '').replace('-con', '-' + appr)
            hexp_ori_stats[appr] = extract_stats_hexp(h_file)

        formal_ori_stats = extract_stats(formal_file, cut2limits)
        formals_ori.append((formal_file, formal_ori_stats))
        hexps_ori.append(hexp_ori_stats)

    """
    produce tables
    """
    metrics = [('errors', 'euclidean'), ('errors', 'manhattan'),
               ('errors2', 'euclidean'), ('errors2', 'manhattan'),
               ('coefs', 'kendalltau'), ('coefs2', 'kendalltau'),
               ('coefs', 'rbo'), ('coefs2', 'rbo')]

    for i, (formal_file, formal_ori_stats) in enumerate(formals_ori):
        hexp_ori_stats = hexps_ori[i]
        if 'pneumoniamnist' in formal_file:
            dtname = 'pneumoniamnist'
            cls = ''
        elif 'mnist' in formal_file:
            dtname = 'mnist'
            cls = '-1,3' if '1,3' in formal_file else '-1,7'
        elif 'cifar' in formal_file:
            dtname = 'cifar-10'
            cls = '-ship,truck'

        if '10,10' in formal_file:
            downscale = '10,10'
        elif '32,32' in formal_file:
            downscale = '32,32'
        else:
            downscale = '28,28'

        if 'origin' in formal_file:
            origin = '-origin'
        else:
            origin = ''

        prefix = f'{dtname}-{downscale}{cls}{origin}'
        saved_dir = '../stats/tables/img/'
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        for metric in metrics:
            filename = '{}{}-{}-{}.pdf'.format(saved_dir, prefix, metric[0], metric[1])
            gnrt_table(filename, formal_ori_stats, hexp_ori_stats, metric)

def gnrt_tab_metrics():
    tab_files = glob.glob('../stats/tab/*.json')
    tab_files = sorted(filter(lambda l: 'shap' in l or 'lime' in l, tab_files))

    hexp_tab_stats = extract_tab_stats(tab_files)

    metrics = [('errors', 'euclidean'), ('errors', 'manhattan'),
               ('errors2', 'euclidean'), ('errors2', 'manhattan'),
               ('coefs', 'kendalltau'), ('coefs2', 'kendalltau'),
               ('coefs', 'rbo'), ('coefs2', 'rbo')]

    saved_dir = '../stats/tables/tab/'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    for metric in metrics:
        filename = '{}tabular-{}-{}.pdf'.format(saved_dir, metric[0], metric[1])
        gnrt_tab_table(filename, hexp_tab_stats, metric)

def gnrt_jit_metrics():
    tab_lr_files = glob.glob('../stats/jit/*.json')
    tab_lr_files = sorted(filter(lambda l: 'formal' not in l, tab_lr_files))

    hexp_tab_lr_stats = extract_tab_stats(tab_lr_files)
    metrics = [('errors', 'euclidean'), ('errors', 'manhattan'),
               ('errors2', 'euclidean'), ('errors2', 'manhattan'),
               ('coefs', 'kendalltau'), ('coefs2', 'kendalltau'),
               ('coefs', 'rbo'), ('coefs2', 'rbo')]

    saved_dir = '../stats/tables/jit/'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    for metric in metrics:
        filename = '{}jit-{}-{}.pdf'.format(saved_dir, metric[0], metric[1])
        gnrt_tab_table(filename, hexp_tab_lr_stats, metric)

#
# =============================================================================
if __name__ == "__main__":
    """
        Generate image metrics
    """
    gnrt_img_metrics()
    exit()