#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
##

import matplotlib.pyplot as plt
import matplotlib
import glob
import json
import numpy as np
import torch
import os
import collections
from functools import reduce
from pysat.examples.hitman import Hitman
import math
import itertools
from rbo import rbo_dict, rbo
from PIL import Image
import cv2
import pandas as pd

def count_expl(pred_id, preds, pred2expls, shape=(28, 28)):
    var_count = collections.defaultdict(lambda: 0)
    for expl in pred2expls[preds[pred_id]]:
        for lit in expl:
            var = abs(lit)
            var_count[var] += 1
    img = np.zeros(shape)

    img = np.expand_dims(img, axis=0)

    img_inst = torch.tensor(img)

    return img_inst, var_count

def count_expl2(pred_id, preds, pred2expls, shape=(28, 28)):
    lit_count = collections.defaultdict(lambda: 0)
    for expl in pred2expls[preds[pred_id]]:
        for lit in expl:
            lit_count[lit] += 1

    img = np.zeros(shape)

    img = np.expand_dims(img, axis=0)

    img_inst = torch.tensor(img)
    return img_inst, lit_count

# white
def visualise2(pred_id, preds, pred2expls, file, base=1, i=None, suffix='', shape=(28, 28), size=11):
    img_inst, lit_count = count_expl2(pred_id, preds, pred2expls, shape)
    m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}
    for lit in lit_count:
        if lit > 0:
            x = (abs(lit) - 1) // img_inst.shape[-1]
            y = (abs(lit) - 1) % img_inst.shape[-2]
            m_3c_instance['R'][0, x, y] = lit_count[lit] #/ base
            # m_3c_instance['G'][0, x, y] = 1
            # m_3c_instance['B'][0, x, y] = 1

    # expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)
    expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R']], 0)
    # print(expl_3c.cpu().numpy().shape)
    a = np.transpose(expl_3c, (1, 2, 0))
    plt.axis("off")
    imshow = plt.imshow(a, cmap='Reds')#) vmin=0, vmax=1)
    for var in lit_count:
        if var > 0:
            x = (abs(var) - 1) // img_inst.shape[-1]
            y = (abs(var) - 1) % img_inst.shape[-2]
            plt.text(y, x, lit_count[var],
                     ha='center', va='center', color='orange', size=size)
    colorbar = plt.colorbar(imshow, location='right')
    # plt.imshow(a)
    saved_dir = './intersection_stats/plots'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    filename = '{0}/{1}_{2}{3}_whiteonly.pdf'.format(saved_dir,
                                                  file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0],
                                                  pred_id, suffix)  # preds[pred_id])

    if i is not None:
        filename = filename.replace('.pdf', '_{}.pdf'.format(i))
    print(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def count_expl3(pred_id, preds, pred2expls, shape=(28, 28)):
    lit_count = collections.defaultdict(lambda: 0)
    for expl in pred2expls[preds[pred_id]]:
        for lit in expl:
            lit_count[lit] += 1

    img = np.zeros(shape)

    img = np.expand_dims(img, axis=0)

    img_inst = torch.tensor(img)
    return img_inst, lit_count

# black
def visualise3(pred_id, preds, pred2expls, file, base=1, i=None, suffix='', shape=(28, 28), size=11, original_img=None):
    img_inst, lit_count = count_expl3(pred_id, preds, pred2expls, shape)
    m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}

    for lit in lit_count:
        if lit < 0:
            x = (abs(lit) - 1) // img_inst.shape[-1]
            y = (abs(lit) - 1) % img_inst.shape[-2]
            m_3c_instance['R'][0, x, y] = lit_count[lit] #/ base # / 255
            # m_3c_instance['G'][0, x, y] = 1
            # m_3c_instance['B'][0, x, y] = 1

    # expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)
    expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R']], 0)
    # print(expl_3c.cpu().numpy().shape)
    a = np.transpose(expl_3c, (1, 2, 0))
    plt.axis("off")
    imshow = plt.imshow(a, cmap='Blues')# vmin=0, vmax=1)
    for var in lit_count:
        if var < 0:
            x = (abs(var) - 1) // img_inst.shape[-1]
            y = (abs(var) - 1) % img_inst.shape[-2]
            plt.text(y, x, lit_count[var],
                     ha='center', va='center', color='orange', size=size)
    colorbar = plt.colorbar(imshow, location='right')
    # plt.imshow(a)
    saved_dir = './intersection_stats/plots'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    filename = '{0}/{1}_{2}{3}_blackonly.pdf'.format(saved_dir,
                                                  file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0],
                                                  pred_id, suffix)
    # preds[pred_id])

    if i is not None:
        filename = filename.replace('.pdf', '_{}.pdf'.format(i))
    print(filename)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def count_expl4(pred_id, preds, pred2expls, shape=(28, 28)):
    var_count = collections.defaultdict(lambda: 0)
    for expl in pred2expls[preds[pred_id]]:
        for lit in expl:
            var = abs(lit)
            if lit > 0:
                var_count[var] += 1
            else:
                var_count[var] -= 1

    # min_count = min(var_count.values())
    # var_count = {var: var_count[var] + min_count for var in var_count}

    img = np.zeros(shape)

    img = np.expand_dims(img, axis=0)

    img_inst = torch.tensor(img)

    return img_inst, var_count

# all
def visualise4(inst_pixels, pred_id, file, base=1, i=None, suffix='', shape=(28, 28), size=11, var_count={},
               ori_instance=None, ori=False):
    #img_inst, var_count = count_expl4(pred_id, preds, pred2expls, shape)
    img = np.zeros(shape)
    img = np.expand_dims(img, axis=0)
    img_inst = torch.tensor(img)

    m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}

    count_ = {}

    # original image
    assert len(inst_pixels) == shape[0] * shape[1]
    if not ori:
        ori_img = np.zeros(shape)
        for lit in inst_pixels:
            x = (abs(lit) - 1) // img_inst.shape[-1]
            y = (abs(lit) - 1) % img_inst.shape[-2]
            ori_img[x, y] = abs(lit) #1 if lit > 0 else 0
    else:
        ori_img = ori_instance.reshape(shape)
        if ori_img.min() < 0:
            # step 1: convert it to [0 ,2]
            np_image = ori_img + 1
            # step 2: convert it to [0 ,1]
            np_image = np_image - np_image.min()
            ori_img = np_image / (np_image.max() - np_image.min())
    ori_img = np.concatenate([np.expand_dims(ori_img, axis=2) for colour in 'RGB'], axis=2)

    # mask
    mask = np.zeros(shape)
    for var in var_count:
        x = (abs(var) - 1) // img_inst.shape[-1]
        y = (abs(var) - 1) % img_inst.shape[-2]
        ## m_3c_instance['G'][0, x, y] = 1
        ## m_3c_instance['B'][0, x, y] = 1
        #print(var_count[var])
        if var > 0:
            m_3c_instance['R'][0, x, y] = abs(var_count[var])  # / base
            count_[var] = var_count[var]
        else:
            m_3c_instance['R'][0, x, y] = abs(var_count[var])  # / base
            count_[abs(var)] = -var_count[var]

        #mask[x, y] = 255 // 3
        #mask[x, y] = abs(var_count[var]) * 255 // 3# * 2 // 3 #// 3
        #mask[x, y] = abs(var_count[var]) * 255 #// 3# * 2 // 3 #// 3
        mask[x, y] = abs(var_count[var]) * 255 * 1.5 #122 #abs(var_count[var]) * 255 #// 3# * 2 // 3 #// 3

    # heatmap
    plt.axis("off")
    #htmap = plt.cm.plasma(m_3c_instance['R'][0])
    #htmap = plt.cm.viridis(m_3c_instance['R'][0])
    htmap = plt.cm.Wistia(m_3c_instance['R'][0])
    #htmap = plt.cm.autumn(m_3c_instance['R'][0])
    #htmap = plt.cm.YlOrBr(m_3c_instance['R'][0])
    #htmap = plt.cm.Oranges(m_3c_instance['R'][0])
    #htmap = plt.cm.Reds(m_3c_instance['R'][0])
    htmap = plt.imshow(htmap)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('./temp.png', bbox_inches='tight', pad_inches=0)

    htmap = Image.open("./temp.png")
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

    saved_dir = './intersection_stats/plots'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    filename = '{0}/{1}_{2}{3}_mix.pdf'.format(saved_dir,
                                               file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0],
                                               pred_id, suffix)
    # preds[pred_id])
    if i is not None:
        filename = filename.replace('.pdf', '_{}.pdf'.format(i))
    # print(filename)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    return filename

# all
def visualise4_bk(pred_id, file, base=1, i=None, suffix='', shape=(28, 28), size=11, var_count={}):
    #img_inst, var_count = count_expl4(pred_id, preds, pred2expls, shape)
    img = np.zeros(shape)
    img = np.expand_dims(img, axis=0)
    img_inst = torch.tensor(img)

    m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}

    count_ = {}
    print(var_count)
    print(img.shape)
    for var in var_count:
        x = (abs(var) - 1) // img_inst.shape[-1]
        y = (abs(var) - 1) % img_inst.shape[-2]
        # m_3c_instance['G'][0, x, y] = 1
        # m_3c_instance['B'][0, x, y] = 1
        if var > 0:
            m_3c_instance['R'][0, x, y] = abs(var_count[var])  # / base
            count_[var] = var_count[var]
        else:
            m_3c_instance['R'][0, x, y] = -abs(var_count[var])  # / base
            count_[abs(var)] = -var_count[var]

    # expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R', 'G', 'B']], 0)
    expl_3c = torch.cat([m_3c_instance[colour] for colour in ['R']], 0)
    # print(expl_3c.cpu().numpy().shape)
    a = np.transpose(expl_3c, (1, 2, 0))
    plt.axis("off")
    #orig_cmap = matplotlib.cm.seismic
    #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shifted')
    #shrunk_cmap = shiftedColorMap(orig_cmap, start=min(var_count.values())/ base, midpoint=0, stop=max(var_count.values())/ base, name='shrunk')
    #imshow = plt.imshow(a, cmap=shrunk_cmap)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    #print('min(var_count.values():', min(var_count.values()))
    #print('max(var_count.values():', max(var_count.values()))
    #divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min(var_count.values())/ base, vcenter=0,
    #                                         vmax=max(var_count.values())/ base)
    #imshow = plt.imshow(a, cmap='seismic', norm=divnorm)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    vmax = abs(max(count_.values(), key=lambda l: abs(l))) if len(count_) > 0 else 1
    vmin = -vmax
    imshow = plt.imshow(a, cmap='seismic', vmin=vmin,
                        vmax=vmax)#ifted_cmap)#'seismic')#'coolwarm')  # cmap='Blues')
    if shape[0] != 28:#(28, 28)
        for var in var_count:
            x = (abs(var) - 1) // img_inst.shape[-1]
            y = (abs(var) - 1) % img_inst.shape[-2]
            plt.text(y, x, '{0:.2f}'.format(var_count[var]), # if 'axp' not in suffix else var_count[var],
                     ha='center', va='center', color='orange', size=size)
    colorbar = plt.colorbar(imshow, location='right')
    # plt.imshow(a)
    saved_dir = './intersection_stats/plots'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    filename = '{0}/{1}_{2}{3}_mix.pdf'.format(saved_dir,
                                            file.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=0)[0],
                                            pred_id, suffix)
    # preds[pred_id])
    if i is not None:
        filename = filename.replace('.pdf', '_{}.pdf'.format(i))
    #print(filename)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename

# attribution plot
def attr_plot(ori_img, i=None, shape=(28, 28), prefix='', suffix='',
              size=11, var_count={}, newfile=None, flip=False, wffa=False):
    img = np.zeros(shape)
    img = np.expand_dims(img, axis=0)
    img_inst = torch.tensor(img)

    #m_3c_instance = {colour: img_inst.clone().detach() for colour in ['R', 'G', 'B']}

    #count_ = {}

    if flip:
        flip_fn = lambda l: 1.0 - l
        ori_img = flip_fn(ori_img)

    a = np.rint(ori_img * 255).astype(np.uint8)
    img = Image.fromarray(a, 'RGB')
    img.save(newfile.rsplit('_', maxsplit=2)[0] + '_ori.png')

    #plt_image = plt.imshow(ori_img)
    #plt.axis("on")
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig(newfile.rsplit('_', maxsplit=2)[0] + '_ori.pdf', bbox_inches='tight', pad_inches=0)
    #plt.close()

    plt.axis("off")
    #ori_img = np.concatenate([np.expand_dims(ori_img, axis=2) for colour in 'RGB'], axis=2)

    # mask
    #mask = np.zeros(shape)
    mask_ = np.zeros(shape)
    if flip:
        if 'pne' in prefix or 'cifar' in prefix:
            #mask = np.full(shape, 255 * 0.2)
            #mask_v = lambda x: abs(x) * 255 * 0.8
            mask = np.full(shape, 255 * 0.2)
            mask_v = lambda x: abs(x) * 255 * 0.8
        else:
            mask = np.full(shape, 255 * 0)
            mask_v = lambda x: abs(x) * 255 * 1
    else:
        if 'pne' in prefix :
            mask = np.full(shape, 255 * 0.2)
            mask_v = lambda x: abs(x) * 255 * 0.8
        elif 'cifar' in prefix:
            mask = np.full(shape, 255 * 0.2)
            mask_v = lambda x: abs(x) * 255 * 0.8
        else:
            mask = np.full(shape, 255 * 0.2)
            mask_v = lambda x: abs(x) * 255 * 0.8
    var_count = {pixel-1: imprt for pixel, imprt in var_count.items()}
    heatmap = np.zeros((shape))
    for var in var_count:
        x = (abs(var) - 1) // img_inst.shape[-1]
        y = (abs(var) - 1) % img_inst.shape[-2]
        if var > 0:
            heatmap[x, y] = abs(var_count[var])  # / base
        else:
            heatmap[x, y] = abs(var_count[var])  # / base

        mask[x, y] += mask_v(var_count[var])
        mask_[x, y] = 1

    # heatmap
    plt.axis("off")
    htmap = plt.cm.plasma(heatmap)
    #htmap = plt.cm.viridis(m_3c_instance['R'][0])
    #htmap = plt.cm.Wistia(m_3c_instance['R'][0])
    #htmap = plt.cm.autumn(m_3c_instance['R'][0])
    #htmap = plt.cm.YlOrBr(m_3c_instance['R'][0])
    #htmap = plt.cm.Oranges(m_3c_instance['R'][0])
    #htmap = plt.cm.Reds(m_3c_instance['R'][0])
    htmap = plt.imshow(htmap)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('./temp.png', bbox_inches='tight', pad_inches=0)

    htmap = Image.open("./temp.png")
    htmap = np.asarray(htmap)
    resize = (shape[0], shape[1])  # , bbb.shape[2])
    htmap = cv2.resize(htmap, resize)

    # deal with 0 attribution heatmap
    non_imprt = set()
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask_[x, y] == 0:
                non_imprt.add(tuple(htmap[x, y]))
                #print(htmap[x, y])
    #assert len(non_imprt) in (0, 2)
    #if len(non_imprt) == 2:
    #    non_imprt = list(filter(lambda l: l != (255, 255, 255, 255), non_imprt))
    #    assert len(non_imprt) == 1
    #    non_imprt = list(non_imprt[0])

    #    for x in range(mask.shape[0]):
    #        for y in range(mask.shape[1]):
    #            if mask_[x, y] == 0:
    #                htmap[x, y] = non_imprt
    #else:
    lits = sorted(var_count, key=lambda l: abs(var_count[l]))
    for lit in lits:
        x = (abs(lit) - 1) // img_inst.shape[-1]
        y = (abs(lit) - 1) % img_inst.shape[-2]
        ht_val = htmap[x, y]
        if ht_val.tolist() != [255, 255, 255, 255]:
            break
    else:
        print('something wrong')
        exit(1)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if htmap[x, y].tolist() == [255, 255, 255, 255]:
                htmap[x, y] = ht_val

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

    if newfile:
        filename = newfile
    else:
        filename = '{}{}.pdf'.format(prefix, suffix)

        if i is not None:
            filename = filename.replace('.pdf', '_{}.pdf'.format(i))
    #filename = filename.replace('.pdf', '.png')
    #print(filename)
    if flip and 'pne' not in prefix:
        plt.axis("on")
    else:
        plt.axis("off")

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    return filename

def normalise(lit2immprt, min_v=0):
    if lit2immprt:
        max_v = abs(max(lit2immprt.values(), key=abs))
        return {lit: (imprt - min_v) / (max_v - min_v) for lit, imprt in lit2immprt.items()}
    else:
        return lit2immprt

def wrong_preds(info):
    # wrong inst 2 correct explanations
    wrong_insts = collections.defaultdict(lambda: [])

    # wrong inst 2 wrong explanations
    wrong_figs = collections.defaultdict(lambda: [])

    for inst in info['stats']:
        imgs = info['stats'][inst]['expl-imgs']
        imgs = list(filter(lambda l: 'nhl_dist.pdf' in l, imgs))
        assert len(imgs) == 2
        rimg = imgs[-1]

        expls = info['stats'][inst]['expls']
        assert len(expls) == 2

        wrong_inst_id = int(rimg.split('mxint_b0_')[-1].split('_')[1])

        # wrong_insts[wrong_inst_id].append((inst, expls[0]))
        wrong_insts[wrong_inst_id].append(expls[0])

        wrong_figs[wrong_inst_id].append(info['stats'][inst]['img-dir'] + rimg)

    return wrong_insts, wrong_figs


def prepare_(wrong_insts, labels):
    pred2expls = collections.defaultdict(lambda: [])


    for wrong_inst_id in wrong_insts:
        expls = wrong_insts[wrong_inst_id]
        label = labels[wrong_inst_id]
        pred2expls[label].extend(expls)

    n = {label: len(pred2expls[label]) for label in pred2expls}
    return pred2expls, n

def prepare(info, model, file=None):
    pred2expls = collections.defaultdict(lambda: [])

    if 'bnn' in file:
        model = 'bnn'
    elif 'bt' in file:
        model = 'bt'
    else:
        model = 'cnn'

    if '1,3' in file:
        cls = '1,3'
    elif '1,7' in file:
        cls = '1,7'
    else:
        cls = 'all'

    if 'pneumoniamnist' not in file:
        pl_file = './preds_labels/{}mnist_{}.json'.format(model + '_' if model != 'cnn' else '',
                                                cls)
    else:
        pl_file = './preds_labels/{}pneumoniamnist.json'.format(model + '_' if model != 'cnn' else '',
                                                         cls)

    xnum = 'xnum' in file

    with open(pl_file, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda l: l.replace('[', '').replace(']', '') \
                     .replace('{', '').replace('}', '').replace(',', '').strip(), lines))
    lines = list(filter(lambda l: len(l) > 0, lines))
    ids = []
    for i, line in enumerate(lines):
        if 'pred' in line:
            ids.append(i)
        elif 'label' in line:
            ids.append(i)
    preds = [int(p) for p in lines[ids[0] + 1: ids[1]]]
    labels = [int(l) for l in lines[ids[1] + 1:]]
    assert len(preds) == len(labels)

    for inst in info['stats']:
        i = int(inst[4:])
        pred = preds[i]
        #pred = info['stats'][inst]['pred']
        if not xnum:
            try:
                expl = info['stats'][inst]['expl']
            except:
                expl = info['stats'][inst]['expls'][0]
            pred2expls[pred].append(expl)
        else:
            expls = info['stats'][inst]['expls']
            pred2expls[pred].extend(expls)

    nn = [len(pred2expls[k])
         for k in sorted(pred2expls, key=lambda l: abs(int(l)))]
    return pred2expls, nn


def prepare2(info, key='expls'):
    pred2expls_ = {}
    #pred2expls = collections.defaultdict(lambda: [])
    #pred = 0
    n = None
    pred = 0

    for inst in info['stats']:
        inst_id = int(inst[4:])
        pred2expls_[inst_id] = {pred: info['stats'][inst][key]}
        if n is None:
            n = len(pred2expls_[inst_id][pred])
    return pred2expls_, n

def prepare3(info):
    pred2expls_ = {}
    pred2coexes_ = {}
    n0 = None
    n1 = None
    pred = 0
    for inst in info['stats']:
        #print(info['stats'][inst]['expl-imgs'])
        #print(info['stats'][inst].keys())
        inst_id = int(inst[4:])
        if len(info['stats'][inst]['expls']) > 0:
            pred2expls_[inst_id] = {pred: info['stats'][inst]['expls']}
        if len(info['stats'][inst]['coexes']) > 0:
            pred2coexes_[inst_id] = {pred: info['stats'][inst]['coexes']}
        if n0 is None:
            try:
                n0 = len(pred2expls_[inst_id][pred])
            except:
                pass
        if n1 is None:
            try:
                n1 = len(pred2coexes_[inst_id][pred])
            except:
                pass
    return pred2expls_, pred2coexes_, n0, n1

def prepare4(info, key='expls'):
    pred2expls_ = {}
    #pred2expls = collections.defaultdict(lambda: [])
    #pred = 0
    ns = []
    pred = 0

    for inst in info['stats']:
        inst_id = int(inst[4:])
        pred2expls_[inst_id] = {pred: info['stats'][inst][key]}
        n = len(pred2expls_[inst_id][pred])
        ns.append(n)
    return pred2expls_, ns

def cut_exps(xps, nof_xps='all'):
    if nof_xps == 'all':
        xps_ = xps
    else:
        dist_len = sorted({len(xp) for xp in xps})
        if len(dist_len) < nof_xps:
            xps_ = xps
        else:
            cut_len = dist_len[nof_xps - 1]
            xps_ = list(filter(lambda l: len(l) <= cut_len, xps))
    return xps_

def cxp_stats(cxps_):
    nof_cxps = len(cxps_)
    all_lits = {lit for lits in cxps_ for lit in lits}
    lit2stats = {}
    for lit in all_lits:
        #print('lit:', lit)
        no_rate = []
        for cxp in cxps_:
            if lit in cxp:
                r = 1 - (1/len(cxp))
            else:
                r = 1
            #print(cxp)
            #print('r:', r)
            #print()
            no_rate.append(r)
        res = 1 - reduce(lambda x, y: x * y, no_rate)
        #print('lit: {}; res: {}'.format(lit, res))
        #print()
        lit2stats[lit] = res
    return lit2stats

def hitting_set(xps):
    hitting_sets = []
    with Hitman(bootstrap_with=xps, htype='sorted') as hitman:
        for hs in hitman.enumerate():
            hitting_sets.append(hs)
    return hitting_sets

def axp_stats(axps_):
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def axp_stats2(axps_):
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1/len(axp)
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def axp_stats3(axps):
    f_dict = collections.defaultdict(lambda : 0)
    for axp in axps:
        for f in axp:
            f_dict[f] += 1 / (len(axp) * len(axps))
    return f_dict

def latex_heatmaps(preds_labels, saved_file, bg, oris, cut, limits, cut_limit, ptypes, xtype,
                   cut_limit_errors_dt, cut_limit_errors2_dt, cut_limit_coefs_dt, cut_limit_coefs2_dt,
                   nof_insts=None, shape=(10,10)):
    nof_insts_ = len(cut_limit[tuple([cut, limits[0]])])
    if nof_insts is None:
        nof_insts = nof_insts_
    else:
        nof_insts = min(nof_insts, nof_insts_)
    insts = sorted(cut_limit[tuple([cut, limits[0]])].keys())[:nof_insts]
    #insts = range(nof_insts) #[17] + [0, 1, 3, 4, 5] + list(range(6, nof_insts+1))
    latex = []
    for inst_id in insts:
        for limit in limits:
            if limit is None:
                continue
            #imgaes
            imgs = []
            imgs.append(oris[inst_id])
            #for ptype in ptypes:
            for hm_type in ['', '2']:
                imgs.append('../../ttnet/' + cut_limit[tuple([cut, limit])][inst_id]['coex' + hm_type])
            for hm_type in ['', '2']:
                imgs.append('../../ttnet/' + cut_limit[tuple([cut, None])][inst_id]['coex' + hm_type])

            #errors
            error = cut_limit_errors_dt[tuple([cut, limit])][inst_id]['coexes']['euclidean']
            error2 = cut_limit_errors2_dt[tuple([cut, limit])][inst_id]['coexes']['euclidean']
            #coefs
            tau = cut_limit_coefs_dt[tuple([cut, limit])][inst_id]['coexes']['kendalltau']
            rbo = cut_limit_coefs_dt[tuple([cut, limit])][inst_id]['coexes']['rbo']
            tau2 = cut_limit_coefs2_dt[tuple([cut, limit])][inst_id]['coexes']['kendalltau']
            rbo2 = cut_limit_coefs2_dt[tuple([cut, limit])][inst_id]['coexes']['rbo']

            #captions
            captions = ['label: {}; pred :{}'.format(preds_labels['labels'][inst_id],
                                                     preds_labels['preds'][inst_id])]
            captions.append('\\footnotesize	dual err {:.2f}  \\\\ tau: {:.2f} rbo: {:.2f}'.format(error, tau, rbo))
            captions.append('\\footnotesize	dual err {:.2f}  \\\\ tau: {:.2f} rbo: {:.2f}'.format(error2, tau2, rbo2))
                                                          #errors[ptypes[2]]))
            captions.append('{} {}'.format('ground' if shape[0] == 10 else '2h',
                                           'axp'))
            captions.append('{} {}'.format('ground' if shape[0] == 10 else '2h',
                                           'axp2'))

            rows = ['\\begin{figure*}[!t]',
                    '\\centering']

            assert len(imgs) == len(captions)

            scale = '{0:.2f}'.format(1 / len(imgs) - 0.01)

            for i, (img, cap) in enumerate(zip(imgs, captions)):
                scale_ = float(scale) - 0.02 if i == 0 else float(scale) + 0.005
                scale_ = str(scale_)
                subfig = ['\\begin{subfigure}[b]{' + scale_ + '\\textwidth}',
                          '  \\centering',
                          '  \\includegraphics[width=0.9\\textwidth]{' + img + '}',
                          '  \\caption{' + cap + '}',
                          '\\end{subfigure}',
                          '%']
                rows.extend(subfig)

            fcap = 'tau: Kendall Tau [-1, 1]; Rank-Biased Overlap [0, 1]; inst {} cutoff {} {} {}'.format(inst_id, cut, limit,
                                                    'bg' if bg else 'no bg')
            rows.append('\\caption{' + fcap + '}')
            rows.append('\\end{figure*}\n')
            latex.append('\n'.join(rows))
        latex.append('\n')
    print(saved_file)
    with open(saved_file, 'w') as f:
        f.write('\n'.join(latex))

def measure_dist(cnt0, cnt1, shape, metric='manhattan', avg=False):
    assert metric in ('euclidean', 'manhattan')
    pixels = set(cnt0.keys()).union(cnt1.keys())
    for p in pixels:
        assert isinstance(p, int)
    cnt0 = {abs(lit): abs(imprt) for lit, imprt in cnt0.items()}
    cnt1 = {abs(lit): abs(imprt) for lit, imprt in cnt1.items()}
    error = {p: abs(cnt0.get(p, 0) - cnt1.get(p, 0)) for p in pixels}
    # error =
    #print(metric)
    #print()
    #print(cnt0)
    #print()
    #print(cnt1)
    #print()
    #print(error)
    #print()
    #print('pixels:', pixels)
    #print()
    if metric == 'euclidean':
        error = math.sqrt(sum([e ** 2 for e in error.values()]))
    else:
        error = sum(error.values())
    #print('error:', error)
    #print()
    if avg:
        return error / (shape[0] * shape[1])
    else:
        return error

def compare_lists(cnt0, cnt_gt, metric='kendall_tau', inst_id=1, p=0.9):
    #if inst_id == 2:
    #    print('cnt_gt:', cnt_gt)
    #print()
    #if metric == 'rbo':
    #    print('rbo')
    #else:
    #    print('tau')
    #print(cnt0)
    #print()
    for lit in cnt0:
        assert isinstance(lit, int)
    for lit in cnt_gt:
        assert isinstance(lit, int)
    cnt0 = {abs(lit): imprt for lit, imprt in cnt0.items()}
    cnt_gt = {abs(lit): imprt for lit, imprt in cnt_gt.items()}
    cnt0_sort = sort_cnt(cnt0, reverse=True)
    #print(cnt0_sort)
    #print()
    #print()
    cnt_gt_sort = sort_cnt(cnt_gt, reverse=True)

    #print(cnt_gt)
    #print()
    #print(cnt_gt_sort)
    #print('cnt_gt:', cnt_gt)
    #print()
    #print('cnt_gt_sort:', cnt_gt_sort)
    #print()

    if metric == 'kendall_tau':
        #Scipy library
        # C: Concordant pairs
        # D: Discordant pairs
        #M = (C - D)
        #tau = M/(C+D) [-1, 1]
        coef = kendalltau(cnt0_sort['pix2rank'].copy(),
                          cnt_gt_sort['pix2rank'].copy())
    elif metric == 'rbo':
        # Rank Biased Overlap
        coef = rank_biased_overlap(cnt0_sort['pix2rank'].copy(),
                                   cnt_gt_sort['pix2rank'].copy(),
                                   p=p)
    #if inst_id == 2:
    #    print('coef:', coef)
    #print()
    #print(coef)

    #kendall tau
    # Rank Biased Overlap (RBO)
    #print('coef:', coef)

    return coef

def sort_cnt(cnt, reverse=True):
    imprt2pix = collections.defaultdict(lambda : [])
    for pix, imprt in cnt.items():
        imprt2pix[imprt].append(pix)

    imprts = sorted(imprt2pix.keys(), reverse=reverse)
    pix2rank = {}
    for i, imprt in enumerate(imprts):
        for pix in imprt2pix[imprt]:
            pix2rank[pix] = i

    return {'imprt2pix': imprt2pix,
            'pix2rank': pix2rank}


    #return [(p, cnt[p]) for p in sorted(cnt.keys(),
    #                                    key=lambda l: cnt[l],
    #                                    reverse=reverse)]

def update_sort(pix2rank0, pix2rank1):
    pix2rank0 = pix2rank0.copy()
    pix2rank1 = pix2rank1.copy()
    pr0_only = set(pix2rank0.keys()).difference(pix2rank1.keys())
    lst_rank0 = max(pix2rank0.values()) if pix2rank0 else -1

    pr1_only = set(pix2rank1.keys()).difference(pix2rank0.keys())
    lst_rank1 = max(pix2rank1.values()) if pix2rank1 else -1

    for e in pr0_only:
        pix2rank1[e] = lst_rank1 + 1

    for e in pr1_only:
        pix2rank0[e] = lst_rank0 + 1

    return pix2rank0, pix2rank1

def rank_biased_overlap(pix2rank0, pix2rank1, p):
    pix2rank0, pix2rank1 = update_sort(pix2rank0, pix2rank1)
    if not pix2rank0 and not pix2rank1:
        coef = 0.0
    else:
        res = rbo_dict(pix2rank0, pix2rank1, p=p)
        coef = res.ext

    return coef

def kendalltau(pix2rank0, pix2rank1):
    pix2rank0, pix2rank1 = update_sort(pix2rank0, pix2rank1)
    all_elements = set(pix2rank0).union(pix2rank1)
    # intersection = set(pix2rank0).intersection(pix2rank1)

    assert len(pix2rank0) == len(pix2rank1) == len(all_elements)

    all_combs = itertools.combinations(all_elements, 2)

    concordants = []
    discordants = []

    for p1, p2 in all_combs:
        k11 = pix2rank0[p1]
        k12 = pix2rank0[p2]

        k21 = pix2rank1[p1]
        k22 = pix2rank1[p2]

        d1 = (k11 - k12)  # > 0
        d2 = (k21 - k22)  # > 0

        if (d1 > 0 and d2 > 0) or (d1 == 0 and d2 == 0) or (d1 < 0 and d2 < 0):
            concordants.append((p1, p2))
        else:
            discordants.append((p1, p2))

    c = len(concordants)
    d = len(discordants)
    if c == 0 and d == 0:
        tau = -1
    else:
        tau = (c - d) / (c + d)
    return tau