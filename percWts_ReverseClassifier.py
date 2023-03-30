#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vibha Viswanathan

Copyright 2021-23 Vibha Viswanathan. All rights reserved.

Script to train a low vs. high F0 classifier for the Reverse context.
Training is done on ambiguous VOT sEEG data, testing on unseen ambiguous VOT sEEG data.
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
from numpy.random import permutation

subjects = ['P21N001','P21N004','P21N007','P21N008','P22N007'] 
nsubs = len(subjects)
infomat = loadmat('./Extracted/datForClassifier.mat')
numtrials_can_test = 50 # no. of Canonical Test epochs
numtrials_rev_test = 50
ntrials = 1300
ntrials_can = round(ntrials/2)
nsamps = 451
n_pca_components = 0.85
Cparam = 0.0001 # inversely proportional to extent of regularization (default = 1)
twin = 21 # time window (in ms or samples) for smoothing

# Stimulus-responsive regions 
subsetRegions = np.asarray(['(HESCHL) R','(HESCHL) L','(STG) R','(STG) L',
    '(STS) R','(STS) L',
    'Temporal-to-Parietal (GapMap) R','(IPL) R',
    '(PostCG) R',
    '(IFG) R','Frontal-II (GapMap) R',
    'Frontal-II (GapMap) L',
    '(POperc) R','(Insula) R','(Insula) L']) 

nSubsetRegions = len(subsetRegions)
nfolds = 10

meanScore_F0revcan = np.zeros((nsubs))
steScore_F0revcan = np.zeros((nsubs))
meanScore_F0revrev = np.zeros((nsubs))
steScore_F0revrev = np.zeros((nsubs))
weights_allsubs = []
pcacomponents_allsubs = []
model_score_revcan = np.zeros((nsubs,nfolds))
model_score_revrev = np.zeros((nsubs,nfolds))
    
for sind in np.arange(nsubs): # iterate over subjects
    pt = subjects[sind]
    allDat = infomat['allDat']
    temp = allDat[0,sind]
    regs = temp['regs']
    regs = regs[0]
    regs = regs[0]
    dat = temp['dat']
    dat = dat[0]
    dat = dat[0]
    isTest = temp['isTest']
    isTest = isTest[0]
    isTest = isTest[0]
    F0 = temp['F0']
    F0 = F0[0]
    F0 = F0[0]
    F0mid = np.unique(F0).mean()
    VOT = temp['VOT']
    VOT = VOT[0]
    VOT = VOT[0]
    # Extract trials with ambiguous VOT (20 ms)
    whichVOTambig = np.squeeze(VOT==20)
    nchans = (np.squeeze(regs).shape)[0]
    # Set up low vs. high F0 Reverse data
    datrev = dat[:,:,ntrials_can:]
    F0rev = np.squeeze(F0[ntrials_can:])
    whichVOTambigRev = whichVOTambig[ntrials_can:] 
    ind = (F0rev < F0mid)
    yrev = np.zeros((ntrials_can))
    yrev[ind] = 1
    yrev[~ind] = 2
    # Find and remove NaN epochs
    nanindsRev = np.isnan((datrev.sum(axis=0).squeeze()).sum(axis=0).squeeze())
    datrev = datrev[:,:,~nanindsRev]
    yrev = yrev[~nanindsRev]
    whichVOTambigRevGoods = whichVOTambigRev[~nanindsRev]
    numtrials_good_rev = np.sum(~nanindsRev) # good trials (not NaN)
    # Set up low vs. high F0 Canonical data
    datcan = dat[:,:,:ntrials_can]
    F0can = np.squeeze(F0[:ntrials_can])
    whichVOTambigCan = whichVOTambig[:ntrials_can]
    ind = (F0can < F0mid)
    ycan = np.zeros((ntrials_can))
    ycan[ind] = 1
    ycan[~ind] = 2
    # Find and remove NaN epochs
    nanindsCan = np.isnan((datcan.sum(axis=0).squeeze()).sum(axis=0).squeeze())
    datcan = datcan[:,:,~nanindsCan]
    ycan = ycan[~nanindsCan]
    whichVOTambigCanGoods = whichVOTambigCan[~nanindsCan]
    numtrials_good_can = np.sum(~nanindsCan) # good trials (not NaN)
    # Select only ambigous VOT data to use for training and testing
    datcan = datcan[:,:,whichVOTambigCanGoods]
    ycan = ycan[whichVOTambigCanGoods]
    nambigCan = whichVOTambigCanGoods.sum()
    datrev = datrev[:,:,whichVOTambigRevGoods]
    yrev = yrev[whichVOTambigRevGoods]
    nambigRev = whichVOTambigRevGoods.sum()
    # Set up common things 
    nambig_can_rev = np.minimum(nambigCan,nambigRev)
    datcan = datcan[:,:,:nambig_can_rev]
    ycan = ycan[:nambig_can_rev]
    datrev = datrev[:,:,:nambig_can_rev]
    yrev = yrev[:nambig_can_rev]
    # Set up classifier predictor matrix
    Xcan = np.empty((nambig_can_rev,0))
    for k in np.arange(nchans):
        predk = np.transpose(np.squeeze(datcan[:,k,:]))
        # Smooth timecourse using savitzky-golay filter
        predk = savgol_filter(predk,twin,3,axis=1)
        Xcan = np.concatenate((Xcan,predk),axis=1) # columns are predictors
    # end
    Xrev = np.empty((nambig_can_rev,0))
    for k in np.arange(nchans):
        predk = np.transpose(np.squeeze(datrev[:,k,:]))
        # Smooth timecourse using savitzky-golay filter
        predk = savgol_filter(predk,twin,3,axis=1)
        Xrev = np.concatenate((Xrev,predk),axis=1) # columns are predictors
    # end
    # PCA for dimensionality reduction
    pca1 = PCA(n_components=n_pca_components,svd_solver='full')
    pca1.fit(Xrev)
    Xcanpca = pca1.transform(Xcan)
    Xrevpca = pca1.transform(Xrev)
    pcacomponents_allsubs.append(pca1.components_)
    trialsPerFold = round(nambig_can_rev/nfolds)
    weights = np.zeros((Xrevpca.shape[1]))
    for k in np.arange(nfolds):
        model = SVC(C=Cparam,kernel='linear')
        strtind = int(k*trialsPerFold)
        stpind = np.minimum(int(strtind+trialsPerFold),nambig_can_rev)
        testinds = np.arange(strtind,stpind)
        # Train with Reverse
        traininds = np.setdiff1d(np.arange(nambig_can_rev),testinds)
        X_train = Xrevpca[traininds,:]
        y_train = yrev[traininds]
        model.fit(X_train, y_train)
        weights = weights + np.squeeze(model.coef_)
        # Test with Canonical
        X_test1 = Xcanpca[testinds,:]
        y_test1 = ycan[testinds]
        y_test1_predicted = model.predict(X_test1) 
        model_score_revcan[sind,k] = (y_test1_predicted == y_test1).mean()
        # Test with Reverse
        X_test2 = Xrevpca[testinds,:]
        y_test2 = yrev[testinds]
        y_test2_predicted = model.predict(X_test2) 
        model_score_revrev[sind,k] = (y_test2_predicted == y_test2).mean()        
    # end
    weights = weights/nfolds
    weights_allsubs.append(weights)
    meanScore_F0revcan[sind] = model_score_revcan[sind,:].mean()
    steScore_F0revcan[sind] = model_score_revcan[sind,:].std()/(nfolds**0.5)
    meanScore_F0revrev[sind] = model_score_revrev[sind,:].mean()
    steScore_F0revrev[sind] = model_score_revrev[sind,:].std()/(nfolds**0.5)
    # Null distribution
    p = 0.5
# end

goodsubs = np.asarray([0,1]) 
ngoodsubs = goodsubs.shape[0]
weights_reg = np.zeros((ngoodsubs,nSubsetRegions))
weights_timereg = np.zeros((ngoodsubs,nsamps,nSubsetRegions))
nchansPerRegion = np.zeros((ngoodsubs,nSubsetRegions))
for sind in goodsubs: 
    pt = subjects[sind]
    allDat = infomat['allDat']
    temp = allDat[0,sind]
    regs = temp['regs']
    regs = regs[0]
    regs = regs[0]
    regs = regs.squeeze()
    nchans = (np.squeeze(regs).shape)[0]
    regs2 = []
    for k in np.arange(nchans):
        regs2.append(regs[k][0])
    #end
    regs2 = np.asarray(regs2)
    weights_timechan = (np.expand_dims(weights_allsubs[sind],1)*
               pcacomponents_allsubs[sind])
    weights_timechan = weights_timechan.sum(axis=0) # sum over PCA components
    weights_timechan = np.reshape(weights_timechan,
                                  (int(weights_timechan.shape[0]/nchans),
                                   nchans))
    weights_timechan = weights_timechan**2.
    weights_chan = weights_timechan.sum(axis=0)
    weights_time = weights_timechan.sum(axis=1)
    # Pool over channels within each region
    for k in np.arange(nSubsetRegions):
        for m in np.arange(nchans):
            loc = regs2[m].find(subsetRegions[k])
            if (loc!=-1):
                nchansPerRegion[sind,k] = nchansPerRegion[sind,k] + 1
                weights_reg[sind,k] = weights_reg[sind,k] + weights_chan[m]
                weights_timereg[sind,:,k] = weights_timereg[sind,:,k] + weights_timechan[:,m]
            #end
        #end
    #end
#end
avg_weights_reg = weights_reg.sum(axis=0)/nchansPerRegion.sum(axis=0)
avg_weights_timereg = weights_timereg.sum(axis=0)/nchansPerRegion.sum(axis=0)

# Remove NaNs in avg_weights (caused by having no channels in a particular
# region across all subjects)
notNanInds = np.where(np.logical_not(np.isnan(avg_weights_reg)))[0]
avg_weights_reg2 = avg_weights_reg[notNanInds]
avg_weights_timereg2 = avg_weights_timereg[:,notNanInds]
subsetRegions2 = subsetRegions[notNanInds]
nSubsetRegions2 = len(subsetRegions2)

# Plot average (over subjects) classifier weights vs. region
sortind = np.argsort(avg_weights_reg2)
pl.figure()
pl.plot(avg_weights_reg2[sortind]/avg_weights_reg2.sum())
pl.xticks(ticks=np.arange(nSubsetRegions2),
          labels=np.asarray(subsetRegions2)[sortind],rotation=90)
pl.ylabel('Low vs. high F0 classifier \n normalized average squared weight')
pl.tight_layout()

# Plot 2-D grid of average (over subjects) classifier weights vs.
# time and brain region
sortind = np.argsort(avg_weights_reg2)
smoothedWts = avg_weights_timereg2/avg_weights_timereg2.sum()
smoothedWts = np.transpose(savgol_filter(smoothedWts,twin,3,axis=0))
pl.figure()
pl.imshow(smoothedWts[sortind,:],aspect='auto')
pl.colorbar(label='Low vs. high F0 classifier \n normalized average squared weight')
pl.yticks(ticks=np.arange(nSubsetRegions2),
          labels=np.asarray(subsetRegions2)[sortind],
          fontsize=6)
pl.xlabel('Time relative to stimulus onset (ms)')
pl.tight_layout()

# Plot average (over subjects) classifier weights vs. time
avg_weights_time = avg_weights_timereg2.sum(axis=1)
pl.figure()
smoothedWts = avg_weights_time/avg_weights_time.sum()
smoothedWts = savgol_filter(smoothedWts,twin,3)
pl.plot(smoothedWts)
pl.xlabel('Time relative to stimulus onset (ms)')
pl.ylabel('Low vs. high F0 classifier \n normalized average squared weight')
pl.tight_layout()

# Save relevant data to MAT file 
mdict = dict(model_score_revcan=model_score_revcan,
             model_score_revrev=model_score_revrev,
             meanScore_F0revcan=meanScore_F0revcan,
             steScore_F0revcan=steScore_F0revcan, 
             meanScore_F0revrev=meanScore_F0revrev,
             steScore_F0revrev=steScore_F0revrev,
             avg_weights_reg2=avg_weights_reg2,
             avg_weights_timereg2=avg_weights_timereg2,
             subsetRegions2=subsetRegions2)
#savemat('RevClassifierResults.mat',mdict)



