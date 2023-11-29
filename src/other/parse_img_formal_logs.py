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
from copy import copy, deepcopy
import torch.nn as nn
import torch
import argparse
import random
from tqdm import tqdm
import numpy as np
import pickle
import glob
import statistics
import json
import matplotlib.pyplot as plt
from heatmap import axp_stats, axp_stats2, cxp_stats, measure_dist, \
    compare_lists, normalise, attr_plot, hitting_set


#
# =============================================================================
def parse_mnist_log(log, explain, classes='all', visual='../visual', batch=100):
    if 'bt_' in log:
        model = 'bt'
        td = 't25d3' if 't25d3' in log else 't50d3'  # (50, 3)
    elif 'bnn_' in log:
        model = 'bnn'
    else:
        model = ''

    if 'pneumoniamnist' in log.lower():
        dtname = 'pneumoniamnist'
    elif 'mnist' in log.lower():
        dtname = 'mnist'
    else:
        print('something wrong')
        exit(1)

    if '14,14' in log:
        downscale = '14,14'
    elif '10,10' in log:
        downscale = '10,10'
    else:
        downscale = '28,28'

    bg = True if '_bg' in log else False
    smallest = True if '_min' in log else False
    xtype = 'abd' if '_con' not in log else 'con'

    xnum = 1
    if 'xnum' in log:
        xnum = log.split('xnum_')[-1].rsplit('.', maxsplit=1)[0].split('_', maxsplit=1)[0].strip()
        if xnum != 'all':
            xnum = int(xnum)

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

    pred_file = './cmpt/bt_inst/bt_{}_{}'.format(dtname, downscale)
    if downscale == '10,10' and 'origin' not in log:
        if dtname == 'mnist':
            pred_file += '_0.46' if classes == '1,3' else '_0.43'
        elif dtname == 'pneumoniamnist':
            pred_file += '_0.16'
        else:
            # todo
            print('something wrong')
            exit(1)

    if dtname == 'mnist':
        pred_file += '_{}'.format(classes)

    if 'origin' in log:
        pred_file += '_origin'

    inst_feats = []

    for i in range(len(insts) - 1):
        explss_.append([])
        expltimess.append([])
        coexess.append([])
        coextimess.append([])
        explss.append([])
        expl_timess.append([])
        dual_explss.append([])
        rtimes.append(False)
        # inst_f = lines[i+1].split('"IF ', maxsplit=1)[0].rsplit(' THEN', maxsplit=1)[0].split(' AND ')
        # inst_feats.append(inst_f)
        for ii in range(insts[i], insts[i + 1]):
            line = lines[ii]
            if 'explaining:' in line:
                inst_feats.append(line.split(':', maxsplit=1)[-1].strip().strip('[]').split(','))
                inst_feats[-1] = list(map(lambda l: float(l.strip().strip("'").rsplit('==')[-1].strip()),
                                          inst_feats[-1]))
            elif 'expl:' in line:
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
            elif '  explanation:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                explss[-1].append(expl)
            elif '  expl time:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                expl_timess[-1].append(rtime)
            elif '  dual explanation:' in line:
                # expl = line.split('"IF ', maxsplit=1)[-1].rsplit(" THEN", maxsplit=1)[0].replace(' != ', ' ==').split(' AND ')
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                dual_explss[-1].append(expl)
            elif '  rtime:' in line:
                rtime = float(line.split(':', maxsplit=1)[-1])
                rtimes[-1] = rtime
                break
        else:
            continue

    def real_expl(inst_f, expl):
        if 'origin' not in log:
            r_expl = []
            for l in expl:
                assert l >= 0
                r_expl.append(inst_f[l])
        else:
            r_expl = [abs(l)+1 for l in expl]
        return r_expl

    # if len(rtimes) == 0:
    #    rtimes.append(3600 * 24)

    # for i in range(len(rtimes)):
    for i in range(len(insts) - 1):
        expls_ = explss_[i]
        expltimes = expltimess[i]
        coexes = coexess[i]
        coextimes = coextimess[i]
        #try:
        #    expls = explss[i]
        #except:
        #    expls = []
        # todo
        #if len(expls) == 0:
        expls = expls_

        expl_times = expl_timess[i]
        #try:
        #    dual_expls = dual_explss[i]
        #except:
        #    dual_expls = []
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
        #if 'mnist' in dtname:
        #    stats['ori-img'] = '../visual/100/{}/ori{}/{}/b_0_{}_ori.pdf'.format(dtname,
        #                                                                         '_{}'.format(downscale) if downscale != '28,28' else '',
        #                                                                         classes,
        #                                                                         inst[4:])
        #else:
        #    print('something wrong')
        #    exit(1)

        #stats['ori-dist-img'] = stats['ori-img'].replace('.pdf', '_dist.pdf')
        #assert (os.path.isfile(stats['ori-img']) and os.path.isfile(stats['ori-dist-img'])),\
        #    '{}\n{}'.format(stats['ori-img'], stats['ori-dist-img'])
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
            stats['rtime'] = 3600 * 2 if downscale == '28,28' else 3600 * 24

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
                stats['len-expl'] = int(downscale.split(',')[0]) ** 2
        try:
            stats['len-dexpl'] = min([len(x) for x in dual_expls])
        except:
            try:
                stats['len-dexpl'] = min([len(x) for x in coexes])
            except:
                stats['len-dexpl'] = int(downscale.split(',')[0]) ** 2

    saved_dir = '../stats/img'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    with open(saved_dir + '/' + conf + '.json', 'w') as f:
        json.dump(info, f, indent=4)


def dt_limit(file, key, limit):
    saved_dir = '{}/{}/{}'.format(file.rsplit('/', maxsplit=1)[0],
                                  key, limit)

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    with open(file, 'r') as f:
        info = json.load(f)
    iscld = 'cld' in file

    if key == 'expl':
        for inst in info['stats']:
            inst_id = int(inst[4:])
            kk = ['expls-', 'expltimes', 'expls', 'expl-times']
            for k in kk:
                info['stats'][inst][k] = info['stats'][inst][k][:limit]

            expltimes = info['stats'][inst]['expltimes']
            nof_expls = len(expltimes)
            if not iscld:
                if limit > len(info['stats'][inst]['expltimes']):
                    continue
                coextimes = list(filter(lambda l: l <= info['stats'][inst]['expltimes'][-1],
                                        info['stats'][inst]['coextimes']))
                nof_coexes = len(coextimes)

                info['stats'][inst]['coexes'] = info['stats'][inst]['coexes'][:nof_coexes]
                info['stats'][inst]['coextimes'] = coextimes
                info['stats'][inst]['dexpls'] = info['stats'][inst]['dexpls'][:nof_coexes]

                if nof_expls > 0 and nof_coexes > 0:
                    info['stats'][inst]['rtime'] = max(info['stats'][inst]['expltimes'][-1],
                                                       info['stats'][inst]['coextimes'][-1])
                else:
                    info['stats'][inst]['rtime'] = 3600 * 24
            else:
                if nof_expls > 0:
                    info['stats'][inst]['rtime'] = info['stats'][inst]['expltimes'][-1]
                else:
                    info['stats'][inst]['rtime'] = 3600 * 24

            info['stats'][inst]['nof-expls'] = nof_expls
            if not iscld:
                info['stats'][inst]['nof-dexpls'] = nof_coexes

            info['stats'][inst]['avgtime'] = round(expltimes[-1] / nof_expls, 4) if nof_expls > 0 else 3600 * 24

            if not iscld:
                info['stats'][inst]['avgdtime'] = round(coextimes[-1] / nof_coexes, 4) if nof_coexes > 0 else 3600 * 4

            info['stats'][inst]['len-expl'] = min(
                [len(x) for x in info['stats'][inst]['expls-']]) if nof_expls > 0 else 10 * 10
            if not iscld:
                info['stats'][inst]['len-dexpl'] = min(
                    [len(x) for x in info['stats'][inst]['coexes']]) if nof_coexes > 0 else 10 * 10

    else:
        assert 'time' in key
        for inst in info['stats']:
            inst_id = int(inst[4:])
            expltimes = list(filter(lambda l: l <= limit,
                                    info['stats'][inst]['expltimes']))
            nof_expls = len(expltimes)
            if not iscld:
                coextimes = list(filter(lambda l: l <= limit,
                                        info['stats'][inst]['coextimes']))
                nof_coexes = len(coextimes)
            info['stats'][inst]['expls-'] = info['stats'][inst]['expls-'][:nof_expls]
            info['stats'][inst]['expltimes'] = expltimes
            if not iscld:
                info['stats'][inst]['coexes'] = info['stats'][inst]['coexes'][:nof_coexes]
                info['stats'][inst]['coextimes'] = coextimes
            info['stats'][inst]['expls'] = info['stats'][inst]['expls'][:nof_expls]
            info['stats'][inst]['expl-times'] = expltimes
            if not iscld:
                info['stats'][inst]['dexpls'] = info['stats'][inst]['dexpls'][:nof_coexes]
                if nof_expls > 0 and nof_coexes > 0:
                    info['stats'][inst]['rtime'] = max(info['stats'][inst]['expltimes'][-1],
                                                       info['stats'][inst]['coextimes'][-1])
                else:
                    info['stats'][inst]['rtime'] = limit
            else:
                if nof_expls > 0:
                    info['stats'][inst]['rtime'] = info['stats'][inst]['expltimes'][-1]
                else:
                    info['stats'][inst]['rtime'] = limit

            info['stats'][inst]['nof-expls'] = nof_expls
            if not iscld:
                info['stats'][inst]['nof-dexpls'] = nof_coexes

            info['stats'][inst]['avgtime'] = round(expltimes[-1] / nof_expls, 4) if nof_expls > 0 else limit
            if not iscld:
                info['stats'][inst]['avgdtime'] = round(coextimes[-1] / nof_coexes, 4) if nof_coexes > 0 else limit

            info['stats'][inst]['len-expl'] = min(
                [len(x) for x in info['stats'][inst]['expls-']]) if nof_expls > 0 else 10 * 10
            if not iscld:
                info['stats'][inst]['len-dexpl'] = min(
                    [len(x) for x in info['stats'][inst]['coexes']]) if nof_coexes > 0 else 10 * 10

    saved_file = '{}/{}'.format(saved_dir, file.rsplit('/', maxsplit=1)[-1])

    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)
    print(saved_file)

def inst2dt_log(files):
    k2logs = collections.defaultdict(lambda: [])
    for file in files:
        k = file.rsplit('_', maxsplit=1)[0]
        k2logs[k].append(file)

    for k, logs in k2logs.items():
        logs.sort(key=lambda l: int(l.rsplit('_', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]))
        cmb_lines = []
        for log in logs:
            with open(log, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if 'inst:' in line:
                    lines = lines[i:]
                    break

            # rtime = list(filter(lambda l: '  rtime:' in l, lines))
            # if len(rtime) == 0:
            #    rtime = list(filter(lambda l: '  expltime:' in l, lines))[-1].replace('expltime:', 'rtime:')
            #    lines.append(rtime + '\n')
            for i, line in enumerate(lines):
                if '  rtime:' in line:
                    lines = lines[:i+1]
                    break
            cmb_lines.extend(lines)
            cmb_lines.append('\n')

        new_file = k + '.log'
        new_file = new_file.replace('cut/', '')
        with open(new_file, 'w') as f:
            f.write(''.join(cmb_lines))

def calculate_metrics():
    for shape in [(28, 28), (10, 10)]: #
        for origin in [True]: #, False]:
            for dtname in ['mnist', 'pneumoniamnist']:#, 'mnist']:
                classes = ['all'] if dtname == 'pneumoniamnist' else ['1,3', '1,7']
                for cls in classes:
                    size = textsize = 6
                    cut2limits = {'time': [10, 30, 60, 120, 300, 600, 1200, 1800, 3600, None]}
                    downscale = '{},{}'.format(shape[0], shape[1])

                    for bg in ['']: #, '-bg']:
                        xtypes = ['con']
                        for xtype in xtypes:
                            # ground truth
                            gt_file = '../stats/img/bt-{}-{}{}-origin-formal{}.json'.format(dtname, downscale, '-' + cls if cls != 'all' else '',
                                                                                            '-' + xtype if xtype else '')
                            print('ground_truth file:', gt_file)
                            groun_truth = {}
                            with open(gt_file, 'r') as f:
                                gt_info = json.load(f)

                            for inst in gt_info['stats']:
                                inst_id = int(inst[4:])

                                expls = gt_info['stats'][inst]['expls-']
                                coexes = gt_info['stats'][inst]['coexes']

                                if 'con' in xtype:
                                    cxps = expls
                                    axps = coexes
                                else:
                                    axps = expls
                                    cxps = coexes

                                gt = {'axp': axp_stats(axps),
                                      'cxp': cxp_stats(cxps)}
                                gt['nor-axp'] = normalise(gt['axp'])
                                gt['nor-cxp'] = normalise(gt['cxp'])
                                gt['axp2'] = axp_stats2(axps)
                                gt['cxp2'] = gt['cxp']
                                gt['nor-axp2'] = normalise(gt['axp2'])
                                gt['nor-cxp2'] = normalise(gt['cxp2'])
                                groun_truth[inst_id] = gt

                            for cut, limits in cut2limits.items():
                                for limit_id, limit in enumerate(limits):
                                    print('cut: {}; limit: {}'.format(cut, limit))

                                    file = gt_file.replace('../stats/img/',
                                                           '../stats/img/{}'.format('{}/{}/'.format(cut, limit) if limit is not None else ''))
                                    print('parsing file:', file)
                                    with open(file, 'r') as f:
                                        info = json.load(f)

                                    for inst in info['stats']:
                                        inst_id = int(inst[4:])
                                        expls = info['stats'][inst]['expls-']
                                        hts = hitting_set(expls)
                                        coexes = info['stats'][inst]['coexes']

                                        if 'con' in xtype:
                                            lit_count_expls = cxp_stats(expls)
                                            lit_count_hts = axp_stats(hts)
                                            lit_count_coexes = axp_stats(coexes)

                                            count_xtypes = ['cxp', 'axp', 'axp']

                                            lit_count_expls2 = lit_count_expls
                                            lit_count_hts2 = axp_stats2(hts)
                                            lit_count_coexes2 = axp_stats2(coexes)
                                        else:
                                            lit_count_expls = axp_stats(expls)
                                            lit_count_hts = cxp_stats(hts)
                                            lit_count_coexes = cxp_stats(coexes)

                                            count_xtypes = ['axp', 'cxp', 'cxp']

                                            lit_count_expls2 = axp_stats2(expls)
                                            lit_count_hts2 = lit_count_hts
                                            lit_count_coexes2 = lit_count_coexes


                                        if inst_id in groun_truth:
                                            gt = groun_truth[inst_id]
                                        else:
                                            gt = {'axp': {}, 'cxp': {}}
                                            break
                                        counts = [lit_count_expls, lit_count_hts,
                                                  lit_count_coexes]
                                        nor_counts = [normalise(lc) for lc in counts]
                                        counts2 = [lit_count_expls2, lit_count_hts2,
                                                   lit_count_coexes2]
                                        nor_counts2 = [normalise(lc) for lc in counts2]
                                        ptypes = ['expl', 'hts', 'coexes']
                                        errors = {}
                                        avg_errors = {}
                                        errors2 = {}
                                        avg_errors2 = {}
                                        coefs = {}
                                        coefs2 = {}
                                        #for cnt_xtype, cnt, ptype in zip(count_xtypes, counts, ptypes):
                                        for pid, (cnt_xtype, cnt, ptype) in enumerate(zip(count_xtypes, nor_counts, ptypes)):
                                            cnt_gt = gt['nor-' + cnt_xtype]
                                            cnt_gt2 = gt['nor-' + cnt_xtype + '2']
                                            cnt2 = nor_counts2[pid]
                                            error = {}
                                            avg_error = {}
                                            error2 = {}
                                            avg_error2 = {}
                                            for metric in ['manhattan', 'euclidean']:
                                                error[metric] = measure_dist(cnt, cnt_gt, shape, metric)
                                                avg_error[metric] = measure_dist(cnt, cnt_gt, shape, metric, avg=True)
                                                error2[metric] = measure_dist(cnt2, cnt_gt2, shape, metric)
                                                avg_error2[metric] = measure_dist(cnt2, cnt_gt2, shape, metric, avg=True)
                                            coefs[ptype] = {}
                                            coefs2[ptype] = {}
                                            for metric in ('kendall_tau', 'rbo'):
                                                coef = compare_lists(cnt, cnt_gt, metric=metric)
                                                coefs[ptype][metric.replace('_', '')] = coef
                                                coef2 = compare_lists(cnt2, cnt_gt2, metric=metric)
                                                coefs2[ptype][metric.replace('_', '')] = coef2
                                            errors[ptype] = error
                                            avg_errors[ptype] = avg_error
                                            errors2[ptype] = error2
                                            avg_errors2[ptype] = avg_error2

                                        info['stats'][inst]['errors'] = errors
                                        info['stats'][inst]['avg-errors'] = avg_errors
                                        info['stats'][inst]['errors2'] = errors2
                                        info['stats'][inst]['avg-errors2'] = avg_errors2
                                        info['stats'][inst]['coefs'] = coefs
                                        info['stats'][inst]['coefs2'] = coefs2
                                        #print(errors)
                                        #print()
                                        #print(avg_errors)
                                        #print()
                                        #print(errors2)
                                        #print()
                                        #print(avg_errors)
                                        #print()
                                        #print(coefs)
                                        #print()
                                        #print(coefs2)
                                        #print()
                                        # original image
                                        #if origin:
                                        #    ori_img = np.stack(np.split(np.array(info['stats'][inst]['inst']),
                                        #                                shape[0]))
                                        #else:
                                        #    ori_img = [1 if lit > 0 else 0 for lit in info['stats'][inst]['inst']]
                                        #    ori_img = np.stack(np.split(np.array(ori_img),
                                        #                                shape[0]))
                                        #if ori_img.min() < 0:
                                        #    # step 1: convert it to [0 ,2]
                                        #    np_image = ori_img + 1
                                        #    # step 2: convert it to [0 ,1]
                                        #    np_image = np_image - np_image.min()
                                        #    ori_img = np_image / (np_image.max() - np_image.min())

                                        #saved_dir = '../plots/{}/'.format(file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0])
                                        #if not os.path.isdir(saved_dir):
                                        #    os.makedirs(saved_dir)

                                        #prefix = saved_dir + file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0]

                                        #heatmaps = {}
                                        #heatmaps['expl'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                             var_count=nor_counts[0], prefix=prefix, suffix='_expl{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #heatmaps['expl2'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                             var_count=nor_counts2[0], prefix=prefix, suffix='_expl2{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #heatmaps['hts'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                            var_count=nor_counts[1], prefix=prefix, suffix='_hts{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #heatmaps['hts2'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                            var_count=nor_counts2[1], prefix=prefix, suffix='_hts2{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #heatmaps['coex'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                             var_count=nor_counts[2], prefix=prefix, suffix='_coex{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #heatmaps['coex2'] = attr_plot(ori_img, i=inst_id, shape=shape, size=size,
                                        #                             var_count=nor_counts2[2], prefix=prefix, suffix='_coex2{}'.format('_{}_{}'.format(cut, limit) if limit is not None else ''))
                                        #info['stats'][inst]['heatmaps'] = heatmaps

                                    # adding errors and heatmaps
                                    with open(file, 'w') as f:
                                        json.dump(info, f, indent=4)

#
# =============================================================================
if __name__ == "__main__":
    """
    merge inst level logs to dt level
    """
    logs = sorted(glob.glob('../logs/img/cut/*.log'),
                  key=lambda l: int(l.replace('cut', '').rsplit('.', maxsplit=1)[0].rsplit('_', maxsplit=1)[1]))
    logs = list(filter(lambda l: 'con' in l, logs))
    inst2dt_log(logs)

    """
    parse img-related logs
    """
    logs = {}

    logs[('formal', '1,3')] = sorted(glob.glob('../logs/img/*1,3*.log'))
    logs[('formal', '1,7')] = sorted(glob.glob('../logs/img/*1,7*.log'))
    logs[('formal', 'all')] = sorted(glob.glob('../logs/img/*pneu*.log'))
    for (explain, classes), logs_ in logs.items():
        for log in sorted(logs_):
            print(log)
            parse_mnist_log(log, classes=classes, explain=explain)

    """
        cut off dt level
    """
    files = sorted(glob.glob('../stats/img/*origin*.json'))
    files = list(filter(lambda l: 'formal' in l, files))
    key2limits = {'time': [10, 30, 60, 120, 300, 600, 1200, 1800, 3600]}
    for file in files:
        for key, limits in key2limits.items():
            for limit in limits:
                print(f'{file=}')
                print(key)
                print(limit)
                print()
                dt_limit(file, key=key, limit=limit)

    """
    calculate metrics
    """
    calculate_metrics()
    exit()
