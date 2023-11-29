#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

# imported modules:
# ==============================================================================
from __future__ import print_function
import collections
import os
import glob
import statistics
import json
import matplotlib.pyplot as plt
from heatmap import attr_plot, normalise, axp_stats, axp_stats2, measure_dist, compare_lists

#
# =============================================================================
def parse_tab_formal_log(log):
    model = 'bt'
    dtname = log.rsplit('/', maxsplit=1)[-1].split('_')[2]
    bg = True if '_bg' in log else False
    smallest = True if '_min' in log else False
    xtype = 'abd' if '_con' not in log else 'con'

    #conf = '{}{}{}{}{}'.format(model + '-' if model != '' else '',
    #                                  dtname,
    #                                  '-bg' if bg else '',
    #                                  '-min' if smallest else '',
    #                                  '-con' if xtype == 'con' else '')

    #conf = conf.replace('--', '-')
    conf = log.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-')
    label = conf
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    lines = open(log, 'r').readlines()

    for i, line in enumerate(lines):
        if 'inst:' in line:
            lines = lines[i:]
            break
    else:
        print('something wrong')
        exit()

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)
    insts.append(len(lines))

    rtimes = []
    # expls in progress
    explss_ = []
    # expls runtimes in progress
    expltimess = []
    # coexes in progress
    coexess = []
    # coexes runtimes in progress
    coextimess = []
    # final expls
    explss = []
    # final expls runtimes
    expl_timess = []
    # final dual expls
    dual_explss = []

    inst_feats = lines[:]
    inst_feats = list(filter(lambda l: 'explaining:' in l, inst_feats))
    inst_feats = [l.replace('"', '').replace('IF ', '').split(':', maxsplit=1)[-1].split(' THEN ')[0].strip('" ').strip().split(' AND ') for l in inst_feats]

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        rtimes.append(False)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'expl:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                explss_[-1].append(expl)
            elif '  expltime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expltimess[-1].append(rtime)
            elif 'coex:' in line:
                try:
                    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                    expl = list(map(lambda l: int(l), expl))
                    coexess[-1].append(expl)
                except:
                    continue
            elif '  coextime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                coextimess[-1].append(rtime)
            #elif '  explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    explss[-1].append(expl)
            elif '  expl time:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expl_timess[-1].append(rtime)
            #elif '  dual explanation:' in line:
            #    # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
            #    expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
            #    expl = list(map(lambda l: int(l), expl))
            #    dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        r_expl = [abs(l) for l in expl]
        return r_expl

    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        try:
            expls = explss[i]
        except:
            expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_

        expl_times = expl_timess[i]
        try:
            dual_expls = dual_explss[i]
        except:
            dual_expls = []
        #todo
        #if len(dual_expls) == 0:
        dual_expls = coexes
        status2 = True if rtimes[i] != False else False
        if 'inst' in lines[insts[i]]:
            inst_line = lines[insts[i]]
        elif 'inst' in lines[insts[i] + 1]:
            inst_line = lines[insts[i] + 1]
        elif 'inst' in lines[insts[i] + 2]:
            inst_line = lines[insts[i] + 2]
        elif 'inst' in lines[insts[i] + 3]:
            inst_line = lines[insts[i] + 3]

        inst = 'inst{0}'.format(inst_line.rsplit(':', maxsplit=1)[-1].strip())
        info['stats'][inst] = info['stats'].get(inst, {})
        stats = info['stats'][inst]
        inst_f = inst_feats[i]
        stats['inst'] = inst_f
        stats['status'] = True
        stats['status2'] = status2
        stats['expls-'] = [real_expl(inst_f, expl) for expl in expls_]
        stats['expltimes'] = expltimes
        stats['coexes'] = [real_expl(inst_f, expl) for expl in coexes]
        stats['coextimes'] = coextimes
        stats['expls'] = expls
        stats['expl-times'] = expl_times
        stats['dexpls'] = dual_expls
        try:
            stats['rtime'] = float(rtimes[i])
        except:
            stats['rtime'] = 3600

        try:
            stats['nof-expls'] = len(expls_)
        except:
            stats['nof-expls'] = len(expls)
        try:
            stats['nof-dexpls'] = max(len(coexes), len(dual_expls))
        except:
            stats['nof-dexpls'] = len(dual_expls)
        try:
            stats['avgtime'] = round(expltimes[-1] / len(expls), 4)
        except:
            stats['avgtime'] = 3600 * 24
        try:
            stats['avgdtime'] = round(coextimes[-1] / len(dual_expls), 4)
        except:
            stats['avgdtime'] = 3600 * 24

        try:
            stats['len-expl'] = min([len(x) for x in expls])
        except:
            try:
                stats['len-expl'] = min([len(x) for x in expls_])
            except:
                stats['len-expl'] = len(inst_feats[i])
        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            try:
                stats['len-dexpl'] = min([len(x) for x in coexes])
            except:
                stats['len-dexpl'] = len(inst_feats[i])

    saved_dir = '../stats/tab'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    with open(saved_dir + '/' + conf + '.json', 'w') as f:
        json.dump(info, f, indent=4)

def parse_tab_hexp_log(log):
    model = 'bt'
    #dtname, appr, _ = log.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=2)
    #assert appr in ('lime', 'shap')
    #conf = '{}-{}'.format(appr, dtname)
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

    saved_dir = '../stats/tab'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    saved_file = '{}/{}.json'.format(saved_dir, label)
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

def calculate_metrics_formal(file):
    with open(file, 'r') as f:
        info = json.load(f)
    dtname = file.rsplit('/', maxsplit=1)[-1].split('-', maxsplit=2)[1]
    for inst in info['stats']:
        #print(inst)
        inst_id = int(inst[4:])
        stats = info['stats'][inst]
        features = stats['inst']
        axps = stats['coexes']
        stats['fid2imprt'] = axp_stats(axps)
        stats['nor-fid2imprt'] = normalise(stats['fid2imprt'])
        stats['fid2imprt2'] = axp_stats2(axps)
        stats['nor-fid2imprt2'] = normalise(stats['fid2imprt2'])
        #saved_dir = '../plots/tab/'
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

    if 'lime' in file:
        appr = 'lime'
    else:
        appr = 'shap'
    dtname = file.rsplit('/', maxsplit=1)[-1].split('-', maxsplit=1)[-1].split('-' + appr, maxsplit=1)[0]

    #ground truth
    gt_file = '../stats/tab/bt-{}-formal-con.json'.format(dtname)
    with open(gt_file, 'r') as f:
        gt_info = json.load(f)

    for inst in gt_info['stats']:
        #print(inst)
        inst_id = int(inst[4:])
        stats = info['stats'][inst]
        gt_stats = gt_info['stats'][inst]
        features = gt_stats['inst']

        #saved_dir = '../plots/hexp/tab/'
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
        #print('cnt_gt :', cnt_gt )
        #print()
        #print('cnt_gt2:', cnt_gt2 )
        #print()
        for metric in ['manhattan', 'euclidean']:
            errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric)
            avg_errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric, avg=True)
            errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric)
            avg_errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric, avg=True)
        #print('errors:', errors)
        #print()
        #print('errors2:', errors2)
        #print()
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
        #print('ceofs:', coefs)
        #print()
        #print('ceofs2:', coefs2)
        #print()
        #exit()
        stats['errors'] = errors
        stats['avg-errors'] = avg_errors
        stats['errors2'] = errors2
        stats['avg-errors2'] = avg_errors2
        stats['coefs'] = coefs
        stats['coefs2'] = coefs2

    with open(file, 'w') as f:
        json.dump(info, f, indent=4)

def attr_plot(features, fid2imprt, prefix='', suffix='', newfile=None, names=None, values=None):
    #print(features)
    #print(len(features))
    #print()
    #print(fid2imprt)
    #print()
    if names and values:
        pass
    else:
        fid2imprt = {abs(int(fid)): abs(imprt) for fid, imprt in fid2imprt.items() if abs(imprt) > 0}
        names = []
        values = []

        for fid in sorted(fid2imprt.keys()):
            names.append(features[fid])
            values.append(fid2imprt[fid])

    #ax = plt.plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)

    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots()
    # Fig size
    fig.set_size_inches(4, 4)

    # Create horizontal bars
    ax.barh(y=names, width=values, alpha=0.4, height=0.3, color=(0.2, 0.4, 0.6, 0.6))#'#86bf91', zorder=2)

    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_visible(False)
    #ax.spines['bottom'].set_position('zero')
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='y', pad=3, labelsize=15)

    for h, v in enumerate(values):
        ax.text(v, h+.18, '{:.2f}'.format(v), color='black', #(0.2, 0.4, 0.6, 0.6),
                fontsize=15)#, fontweight='bold')
    #ax.set_axis_off()
    #plt.yticks()
    #print('aaa')
    #exit()
    #plt.figure(figsize=(10, 6))
    #plt.bar(range(len(fid2imprt.keys())), values, tick_label=names, color=(0.2, 0.4, 0.6, 0.6))
    #plt.xticks(rotation=45, ha='right')
    #plt.ylabel('Attribution')
    ##a = plt.show()
    if newfile:
        filename = newfile
    else:
        filename = '{}{}.pdf'.format(prefix, suffix)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #print(' attribution plot saved to', filename)
    plt.close()
    return filename

def latex(hexp_files, dtname, nof_insts=10):
    #for file in hexp_files:
    infos = {file.rsplit('/', maxsplit=1)[-1].split('-', maxsplit=1)[0]:
                 json.load(open(file, 'r'))
             for file in hexp_files}
    gt_file = '../stats/bt-{}_min-con.json'.format(dtname)
    with open(gt_file, 'r') as f:
        gt_info = json.load(f)

    insts = sorted(infos['lime']['stats'].keys(), key=lambda l: int(l[4:]))
    for inst in insts:
        inst_id = int(inst[4:])
        imgs = []
        captions = []
        for appr in infos:
            nor_heatmap = infos[appr]['stats'][inst]['heatmaps']['nor-attr']
    print(insts)
    exit()

#
# =============================================================================
if __name__ == "__main__":
    """
    parse formal tabular data logs
    """
    logs = sorted(glob.glob('../logs/tab/*.log'))
    for log in logs:
        print(log)
        parse_tab_formal_log(log)

    files = sorted(glob.glob('../stats/tab/*formal*.json'))
    for file in files:
        print(file)
        calculate_metrics_formal(file)

    """
    parse tabular data hexp logs
    """
    logs = sorted(glob.glob('../logs/hexp/tab/*.log'))
    for log in logs:
        print(log)
        parse_tab_hexp_log(log)

    files = glob.glob('../stats/tab/*.json')
    files = sorted(filter(lambda l: 'lime' in l or 'shap' in l,
                          files))
    for file in files:
        print(file)
        calculate_metrics_hexp(file)
    exit()