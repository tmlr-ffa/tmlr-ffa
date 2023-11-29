#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
##

#
#==============================================================================
from __future__ import print_function
import collections
import matplotlib.pyplot as plt

def normalise(lit2immprt, min_v=0):
    if lit2immprt:
        max_v = abs(max(lit2immprt.values(), key=abs))
        return {lit: (imprt - min_v) / (max_v - min_v) for lit, imprt in lit2immprt.items()}
    else:
        return lit2immprt

def cal_ffa(axps_):
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def cal_wffa(axps_):
    lit_count = collections.defaultdict(lambda: 0)
    nof_axps = len(axps_)
    for axp in axps_:
        for lit in axp:
            lit_count[lit] += 1/len(axp)
    lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
    return lit_count

def attr_plot(features, fid2imprt, newfile=None, names=None, values=None):
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


    if newfile:
        filename = newfile
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        #print(' attribution plot saved to', filename)
    else:
        plt.show()

    plt.close()

