#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vibha Viswanathan

Copyright 2021-23 Vibha Viswanathan. All rights reserved.

Script to compare classifier performance between the Canonical and Reverse classifiers.
"""

import numpy as np
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.signal import lfilter, savgol_filter
import timeit
from numpy.random import permutation, binomial

subjects = ['P21N001','P21N004','P21N007','P21N008','P22N007'] 
nsubs = len(subjects)
nfolds = 10
twin = 21 # time window (in ms or samples) for smoothing
resultsdir = ('./')
dictCan = loadmat(resultsdir + 'CanClassifierResults.mat')
dictRev = loadmat(resultsdir + 'RevClassifierResults.mat')

score_cancan = dictCan['model_score_cancan'].squeeze() 
score_canrev = dictCan['model_score_canrev'].squeeze()
avg_weights_reg2_can = dictCan['avg_weights_reg2'].squeeze()
avg_weights_timereg2_can = dictCan['avg_weights_timereg2'].squeeze()
meanScore_F0cancan = dictCan['meanScore_F0cancan'].squeeze()
steScore_F0cancan = dictCan['steScore_F0cancan'].squeeze()
meanScore_F0canrev = dictCan['meanScore_F0canrev'].squeeze()
steScore_F0canrev = dictCan['steScore_F0canrev'].squeeze()
subsetRegions2temp = dictCan['subsetRegions2'].squeeze()
nSubsetRegions2 = len(subsetRegions2temp)
subsetRegions2 = np.empty((nSubsetRegions2), dtype='object')
for k in np.arange(nSubsetRegions2):
    subsetRegions2[k] = subsetRegions2temp[k].strip()
#end
score_revcan = dictRev['model_score_revcan'].squeeze()
score_revrev = dictRev['model_score_revrev'].squeeze()
avg_weights_reg2_rev = dictRev['avg_weights_reg2'].squeeze()
avg_weights_timereg2_rev = dictRev['avg_weights_timereg2'].squeeze()
meanScore_F0revrev = dictRev['meanScore_F0revrev'].squeeze()
steScore_F0revrev = dictRev['steScore_F0revrev'].squeeze()

# Pool data over all subjects
score_cancan = np.reshape(score_cancan,(nsubs*nfolds))
score_canrev = np.reshape(score_canrev,(nsubs*nfolds))
score_revcan = np.reshape(score_revcan,(nsubs*nfolds))
score_revrev = np.reshape(score_revrev,(nsubs*nfolds))
score_diff1 = score_cancan - score_canrev
score_diff2 = score_cancan - score_revrev

# Derive permutation-based null distributions for score differences
niters = 10000 # number of iterations
pBer = 0.5 # Bernouilli trial p
null1 = np.zeros((niters))
null2 = np.zeros((niters))
for k in np.arange(niters):
    # Randomly flip the sign of the score difference in each fold
    flipOrNot = binomial(1,pBer,nsubs*nfolds) # generate nfolds Bernouilli trials
    flipOrNot[flipOrNot==0] = -1
    null1[k] = (flipOrNot*score_diff1).mean()
    null2[k] = (flipOrNot*score_diff2).mean()
#end
# Derive p-values for actual score differences 
mean_score_diff1 = score_diff1.mean()
mean_score_diff2 = score_diff2.mean()
pval1 = ((null1 >= mean_score_diff1).sum())/niters 
pval2 = ((null2 >= mean_score_diff2).sum())/niters

# Plot classifier performance pooled over subjects
colsetdark = ['#1b9e77','#d95f02','#7570b3']
colsetlight = ['#66c2a5','#fc8d62','#8da0cb']
colset = colsetlight
col1 = colset[0]
col2 = colset[1]
col3 = colset[2]
if (pval1 == 0):
    pvalstr1 = (' p < ' + 
     str(np.format_float_positional((1/niters),unique=False, precision=4)))
else:
    pvalstr1 = (' p = ' + 
     str(np.format_float_positional(pval1,unique=False, precision=4)))
#end
if (pval2 == 0):
    pvalstr2 = (' p < ' + 
     str(np.format_float_positional((1/niters),unique=False, precision=4)))
else:
    pvalstr2 = (' p = ' + 
     str(np.format_float_positional(pval2,unique=False, precision=4)))
#end
pl.figure()
pchance = 0.5 # chance level
mean_bar1 = score_cancan.mean()
ste_bar1 = score_cancan.std()/np.sqrt(nsubs*nfolds)
mean_bar2 = score_canrev.mean()
ste_bar2 = score_canrev.std()/np.sqrt(nsubs*nfolds)
mean_bar3 = score_revrev.mean()
ste_bar3 = score_revrev.std()/np.sqrt(nsubs*nfolds)
barxloc = [1,1.5]
ylim1 = [0.47,0.88]
yticks1 = np.arange(0.5,0.9,0.1)
whichChance = np.where(yticks1==pchance)[0][0]
for k in np.arange(yticks1.shape[0]):
    yticks1[k] = np.format_float_positional(yticks1[k],unique=False,precision=1)
#end
yticklabelstr = yticks1.astype(str)
yticklabelstr[whichChance] = yticklabelstr[whichChance] + '\n(chance)'
# Subplot 1
pl.subplot(1,2,1)
pl.ylabel('Proportion correct',fontsize=14)
pl.bar(barxloc[0],mean_bar1,color=col1,width=0.3)
pl.bar(barxloc[1],mean_bar2,color=col2,width=0.3)
pl.errorbar(barxloc[0],mean_bar1,ste_bar1,color='k')
pl.errorbar(barxloc[1],mean_bar2,ste_bar2,color='k')
pl.xticks(barxloc)
ax = pl.gca()
ax.set_xticklabels(['Train Canonical,\nTest Canonical',
                    'Train Canonical,\nTest Reverse'],fontsize=14)
pl.ylim(ylim1)
pl.yticks(yticks1)
ax.set_yticklabels(yticklabelstr,fontsize=14)
pl.text(1.3,0.85,pvalstr1,fontsize=12)
for k in np.arange(nsubs):
    pl.plot(barxloc,[meanScore_F0cancan[k],meanScore_F0canrev[k]],color=[0.75,0.75,0.75])
#end
# Subplot 2
pl.subplot(1,2,2)
pl.bar(barxloc[0],mean_bar1,color=col1,width=0.3)
pl.bar(barxloc[1],mean_bar3,color=col3,width=0.3)
pl.errorbar(barxloc[0],mean_bar1,ste_bar1,color='k')
pl.errorbar(barxloc[1],mean_bar3,ste_bar3,color='k')
pl.xticks(barxloc)
ax = pl.gca()
ax.set_xticklabels(['Train Canonical,\nTest Canonical',
                    'Train Reverse,\nTest Reverse'],fontsize=14)
pl.ylim(ylim1)
pl.yticks(yticks1)
pl.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=True,      # ticks along the left edge are on
    labelleft=False) # labels along the left edge are off
pl.text(1.3,0.85,pvalstr2,fontsize=12)
for k in np.arange(nsubs):
    pl.plot(barxloc,[meanScore_F0cancan[k],meanScore_F0revrev[k]],color=[0.75,0.75,0.75])
#end





