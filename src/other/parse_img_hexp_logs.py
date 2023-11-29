#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

import csv
import pandas as pd
import pickle
import glob
import json
from heatmap import attr_plot, normalise, axp_stats, axp_stats2, measure_dist, compare_lists
import collections
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def parse_log(file, appr):
    with open(file, 'r') as f:
        lines = f.readlines()

    model = 'bt' if '/bt/' in file else 'bnn'
    downscale = '10,10' if '10,10' in file else '28,28'
    shape = tuple([int(l) for l in downscale.split(',')])
    isorigin = 'origin' in file
    dtname = file.rsplit('/', maxsplit=1)[-1].split('_', maxsplit=1)[0]
    if dtname == 'mnist':
        cls = '1,3' if '1,3' in file else '1,7'
    else:
        cls = 'all'

    #label = '{}-{}-{}{}-{}{}'.format(model, appr, dtname,
    #                                 '-origin' if isorigin else '',
    #                                 downscale,
    #                                 '-' + cls if cls != 'all' else '')
    label = file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0].replace('_', '-')
    info = {"preamble": {"program": label, "prog_args": label, "prog_alias": label, "benchmark": None},
            "stats": {}}

    insts = []
    for i, line in enumerate(lines):
        if 'inst:' in line:
            insts.append(i)

    insts.append(len(lines))
    for inst_id, i in enumerate(insts[:-1]):
        info['stats']['inst{}'.format(inst_id)] = {}
        stats = info['stats']['inst{}'.format(inst_id)]
        for j in range(i+1, insts[inst_id+1]):
            line = lines[j]
            #if 'saved to' in line:
            #    stats['heatmap'] = line.rsplit(maxsplit=1)[-1]
            if '  explaining:' in line:
                inst = line.split(':', maxsplit=1)[-1].split(' AND ')
                inst = list(map(lambda l: float(l.split(' = ')[-1]), inst))
                stats['inst'] = inst
            elif '  expl:' in line:
                expl = line.split(':', maxsplit=1)[-1].strip().strip('[]').split(',')
                expl = list(map(lambda l: int(l), expl))
                stats['expl'] = expl
            elif '  lit2imprt:' in line:
                lit2imprt = line.split(':', maxsplit=1)[-1].strip().strip('{}').split(',')
                lit2imprt = map(lambda l: l.split(':'), lit2imprt)
                lit2imprt = {abs(int(l[0])): float(l[1]) for l in lit2imprt}
                stats['lit2imprt'] = lit2imprt
                stats['nor-lit2imprt'] = normalise(lit2imprt, min_v=0)
                #stats['pos-lit2imprt'] = {lit: imprt for lit, imprt in stats['lit2imprt'].items() if imprt > 0}
                #stats['nor-pos-lit2imprt'] = normalise(stats['pos-lit2imprt'], min_v=0)

            elif '  time:' in line:
                time = float(line.split(':', maxsplit=1)[-1])
                stats['rtime'] = time

        #if 'mnist' in dtname:
        #    stats['ori-img'] = '../visual/100/{}/ori{}/{}/b_0_{}_ori.pdf'.format(dtname,
        #                                                                         '_{}'.format(downscale) if downscale != '28,28' else '',
        #                                                                         cls,
        #                                                                         inst_id)
        #else:
        #    #todo
        #    print('something wrong')
        #    print('tabular data')
        #    exit(1)

        #stats['ori-dist-img'] = stats['ori-img'].replace('.pdf', '_dist.pdf')


        #assert (os.path.isfile(stats['ori-img']) and os.path.isfile(stats['ori-dist-img'])), \
        #    '{}\n{}'.format(stats['ori-img'], stats['ori-dist-img'])


    saved_file = '../stats/img/{}.json'.format(label)
    with open(saved_file, 'w') as f:
        json.dump(info, f, indent=4)

def calculate_metrics(file, appr):
    with open(file, 'r') as f:
        info = json.load(f)

    model = 'bt' if '/bt/' in file else 'bnn'
    downscale = '10,10' if '10,10' in file else '28,28'
    shape = tuple([int(l) for l in downscale.split(',')])
    for dtname in ['pneumoniamnist', 'mnist']:
        if dtname in file:
            break
    else:
        print('unknown data')
        exit(1)

    if dtname == 'mnist':
        cls = '1,3' if '1,3' in file else '1,7'
    else:
        cls = 'all'

    # ground truth
    xtype = 'con'
    gt_file = '../stats/img/bt-{}-{}{}-origin-formal{}.json'.format(dtname, downscale, '-' + cls if cls != 'all' else '', '-' + xtype if xtype else '')
    print('groung_truth file:', gt_file)
    groun_truth = {}
    with open(gt_file, 'r') as f:
        gt_info = json.load(f)

    gt_imgs = {}
    for inst in gt_info['stats']:
        inst_id = int(inst[4:])

        expls = gt_info['stats'][inst]['expls-']
        coexes = gt_info['stats'][inst]['coexes']

        if 'con' in xtype:
            #cxps = expls
            axps = coexes
        else:
            axps = expls
            #cxps = coexes

        gt = {}
        #gt['axp']: axp_stats(axps)}
        gt['nor-axp'] = normalise(axp_stats(axps))
        #gt['axp2'] = axp_stats2(axps)
        gt['nor-axp2'] = normalise(axp_stats2(axps))
        groun_truth[inst_id] = gt
        #gt_imgs[inst_id] = {'nor-axp': gt_info['stats'][inst]['heatmaps']['coex'],
        #                    'nor-axp2': gt_info['stats'][inst]['heatmaps']['coex2']}

    for inst in gt_info['stats']:
        inst_id = int(inst[4:])
        #info['stats'][inst]['gt-imgs'] = gt_imgs[inst_id]

        cnt_gt = groun_truth[inst_id]['nor-axp'] # avg of sum of counts
        cnt_gt2 = groun_truth[inst_id]['nor-axp2'] # avg of sum of fractions
        cnt_hexp = {abs(int(lit)): abs(imprt) for lit, imprt in info['stats'][inst]['lit2imprt'].items()}
        cnt_hexp_nor = normalise(cnt_hexp)
        #print(cnt_gt)
        #print()
        #print(cnt_gt2)
        #print()
        #print(cnt_hexp)
        #print()
        #print(cnt_hexp_nor)
        #print()

        errors = {}
        errors2 = {}
        avg_errors = {}
        avg_errors2 = {}
        coefs = {}
        coefs2 = {}

        for metric in ['manhattan', 'euclidean']:
            errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric)
            avg_errors[metric] = measure_dist(cnt_hexp_nor, cnt_gt, shape, metric, avg=True)
            errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric)
            avg_errors2[metric] = measure_dist(cnt_hexp_nor, cnt_gt2, shape, metric, avg=True)
        #print(errors)
        #print()
        #print(avg_errors)
        #print()
        #print(errors2)
        #print()
        #print(avg_errors2)
        #print()
        for metric in ('kendall_tau', 'rbo'):
            coef = compare_lists(cnt_hexp_nor, cnt_gt, metric=metric)
            coefs[metric.replace('_', '')] = coef
            coef2 = compare_lists(cnt_hexp_nor, cnt_gt2, metric=metric)
            coefs2[metric.replace('_', '')] = coef2
        #print(coefs)
        info['stats'][inst]['errors'] = errors
        info['stats'][inst]['avg-errors'] = avg_errors
        info['stats'][inst]['errors2'] = errors2
        info['stats'][inst]['avg-errors2'] = avg_errors2
        info['stats'][inst]['coefs'] = coefs
        info['stats'][inst]['coefs2'] = coefs2
        #print()
        #print(coefs2)
        # original image
        #ori_img = np.stack(np.split(np.array(info['stats'][inst]['inst']),
        #                            shape[0]))
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
        #heatmap = attr_plot(ori_img, i=inst_id, shape=shape,
        #                             var_count=cnt_hexp_nor, prefix=prefix)
        #info['stats'][inst]['heatmap'] = heatmap

    with open(file, 'w') as f:
        json.dump(info, f, indent=4)

def visualise(instance, saved_file, lit2imprt, shape=(10, 10), original_img=None):
    img = np.zeros(shape)
    img = np.expand_dims(img, axis=0)
    img_inst = torch.tensor(img)

    m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}
    count_ = {}

    # original image
    assert len(instance) == shape[0] * shape[1]
    if not original_img:
        ori_img = instance.reshape(shape)
    else:
        ori_img = instance.reshape(shape)
        if ori_img.min() < 0:
            # step 1: convert it to [0 ,2]
            np_image = ori_img + 1
            # step 2: convert it to [0 ,1]
            np_image = np_image - np_image.min()
            ori_img = np_image / (np_image.max() - np_image.min())
    ori_img = np.concatenate([np.expand_dims(ori_img, axis=2) for colour in 'RGB'], axis=2)

    # mask
    mask = np.zeros(shape)
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

        mask[x, y] = abs(imprt) * 255 * 1.5 #// 1.5 # 255 // 5 #abs(imprt) * 255 // 2 #2 // 3#// 2

    # heatmap
    plt.axis("off")
    #htmap = plt.cm.Oranges(m_3c_instance['R'][0])
    htmap = plt.cm.Wistia(m_3c_instance['R'][0])
    htmap = plt.imshow(htmap)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('./temp2.png', bbox_inches='tight', pad_inches=0)

    htmap = Image.open("./temp2.png")
    htmap = np.asarray(htmap)

    resize = (shape[0], shape[1])  # , bbb.shape[2])
    htmap = cv2.resize(htmap, resize)

    # final
    # background
    background = ori_img * 255
    background = background.astype(np.uint8)
    background = Image.fromarray(background)

    # top image
    htmap = Image.fromarray(htmap)

    # mask
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)

    # final heatmap
    final = background.copy()
    final.paste(htmap, (0, 0), mask)

    a = plt.imshow(np.asarray(final))

    saved_dir = saved_file.rsplit('/', maxsplit=1)[0]
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    #print('  heatmap saved to', saved_file)
    plt.savefig(saved_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    ## expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)
    #expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R']], 0)
    ## print(expl_3c.cpu().numpy().shape)
    #a = np.transpose(expl_3c, (1, 2, 0))
    #plt.axis("off")
    ## orig_cmap = matplotlib.cm.seismic
    ## shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shifted')
    ## shrunk_cmap = shiftedColorMap(orig_cmap, start=min(var_count.values())/ base, midpoint=0, stop=max(var_count.values())/ base, name='shrunk')
    ## imshow = plt.imshow(a, cmap=shrunk_cmap)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    ## print('min(var_count.values():', min(var_count.values()))
    ## print('max(var_count.values():', max(var_count.values()))
    ## divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min(var_count.values())/ base, vcenter=0,
    ##                                         vmax=max(var_count.values())/ base)
    ## imshow = plt.imshow(a, cmap='seismic', norm=divnorm)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    #vmax = abs(max(count_.values(), key=lambda l: abs(l))) if len(count_) > 0 else 1
    #vmin = -vmax
    #imshow = plt.imshow(a, cmap='seismic', vmin=vmin,
    #                    vmax=vmax)  # ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    ## for lit in lit2imprt:
    ##    x = (abs(lit) - 1) // img_inst.shape[-1]
    ##    y = (abs(lit) - 1) % img_inst.shape[-2]
    ##    plt.text(y, x, '{0:.2f}'.format(lit2imprt[lit]),  # if 'axp' not in suffix else var_count[var],
    ##             ha='center', va='center', color='orange', size=size)
    #colorbar = plt.colorbar(imshow, location='right')
    ## plt.imshow(a)

    ##print('  heatmap saved to', saved_file)
    #plt.savefig(saved_file, bbox_inches='tight', pad_inches=0)
    #plt.close()

def latex(conf, files, preds_labels, shape, nor='', pos='', nof_insts=10):
    insts = range(nof_insts)#[0, 1, 3, 4, 5] + list(range(6, nof_insts+1))
    latex = []

    #['ori', 'ori-dist', 'lime', 'shap', 'gt']
    #['ori', 'ori-dist', 'lime', 'shap']

    infos = [json.load(open(file, 'r')) for file in files]

    for inst_id in insts:
        inst = 'inst{}'.format(inst_id)

        imgs = ['../' + infos[0]['stats'][inst]['ori-img']]#, '../' + infos[0]['stats'][inst]['ori-dist-img']]
        caps = ['label: {}; pred:{}'.format(preds_labels['labels'][inst_id], preds_labels['preds'][inst_id])]
        for info in infos:
            imgs.append('../' + info['stats'][inst]['heatmap'])
            if 'lime' in info['preamble']['prog_alias']:
                appr = 'lime'
            elif 'shap' in info['preamble']['prog_alias']:
                appr = 'shap'
            else:
                appr = 'anchor'
            m_error = '{:.2f}'.format(info['stats'][inst]['errors']['euclidean'])
            m_error2 = '{:.2f}'.format(info['stats'][inst]['errors2']['euclidean'])
            tau = info['stats'][inst]['coefs']['kendalltau']
            rbo = info['stats'][inst]['coefs']['rbo']
            tau2 = info['stats'][inst]['coefs2']['kendalltau']
            rbo2 = info['stats'][inst]['coefs2']['rbo']
            caps.append('\\footnotesize	{} {}\\\\ tau: {} rbo: {}'.format(appr,
                                                                             'err: {}, {} '.format(m_error, m_error2),
                                                                             '{:.2f}, {:.2f}'.format(tau, tau2),
                                                                             '{:.2f}, {:.2f}'.format(rbo, rbo2)))

        try:
            imgs.append('../' + info['stats'][inst]['gt-imgs']['nor-axp'])
            imgs.append('../' + info['stats'][inst]['gt-imgs']['nor-axp2'])
            if shape[0] == 28:
                caps.append('\\footnotesize Formal \\\\ 2h AXp')
                caps.append('\\footnotesize 2h AXp2')
            else:
                caps.append('\\footnotesize Formal \\\\ Ground AXp')
                caps.append('\\footnotesize Ground AXp2')
        except:
            pass


        scale = '{0:.2f}'.format(1 / len(imgs) - 0.01)
        rows = ['\\begin{figure*}[!t]',
                '\\centering']

        for i, (img, cap) in enumerate(zip(imgs, caps)):
            scale_ = float(scale) - 0.02 if i == 0 else float(scale) + 0.005
            scale_ = str(scale_)
            subfig = ['\\begin{subfigure}[b]{' + scale_ + '\\textwidth}',
                      '  \\centering',
                      '  \\includegraphics[width=0.9\\textwidth]{' + img + '}',
                      '  \\caption{' + cap + '}',
                      '\\end{subfigure}',
                      '%']
            rows.extend(subfig)

        fcap = 'tau: Kendall Tau [-1, 1]; Rank-Biased Overlap [0, 1]; inst {}'.format(inst_id)
        rows.append('\\caption{' + fcap + '}')
        rows.append('\\end{figure*}\n')
        latex.append('\n'.join(rows))
        latex.append('\n')

    saved_file = conf + '{}{}.tex'.format('-nor' if nor else '',
                                          '-pos' if pos else '')
    saved_file = saved_file.replace('.json', '')
    print(saved_file)
    with open(saved_file, 'w') as f:
        f.write('\n'.join(latex))

if __name__ == '__main__':
    files = sorted(glob.glob('../logs/hexp/img/*.log'))
    for file in files:
        for appr in ['lime', 'shap']:
            if appr in file:
                break
        else:
            print('something wrong')
            exit(1)
        print(file)
        parse_log(file, appr)

    for appr in ['lime', 'shap']: #, 'anchor']
        files = sorted(glob.glob('../stats/img/*{}*.json'.format(appr)))
        for file in files:
            print(file)
            calculate_metrics(file, appr)
    exit()
