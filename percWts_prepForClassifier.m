% Script to save data in a format suitable for the classifier analysis.
%
% Copyright 2021-23 Vibha Viswanathan. All rights reserved.

subjects = {'P21N001','P21N004','P21N007','P21N008','P22N007'}; 
nsubs = numel(subjects);
task = 'percWts'; 
datarootpath = '/OneDrive/1. Recording & Stimulation of iEEG';
addpath(genpath(fullfile(datarootpath,'code')));
extractedpath = fullfile('./Extracted'); % location of pre-processed data
fs_EEG = 1000; % in Hz
stimDur = 250; % in ms
epochWin_sec = [-100 stimDur+200]/1000; % epoch window relative to stim onset (in seconds)
subsetRegions = {'(HESCHL) L','(STS) L','Temporal-to-Parietal (GapMap) L','(HESCHL) R','(STG) R','(STG) L',...
     '(OFC) R','(STS) R','Temporal-to-Parietal (GapMap) R','(IPL) R','(POperc) R','(Insula) R',...
     '(IPL) L','(PostCG) R','(Insula) L','Frontal-II (GapMap) L','Frontal-II (GapMap) R',...
     '(IFG) R','(SFS) L','(Amygdala) R','(PostCG) L','(IFG) L','(PostCS) L',...
     '(IPS) R','(IPS) L','(SPL) L','(PreCG) R'}; % non-visual stimulus responsive regions (based on latency analysis)

for sind = 1:nsubs % iterate over subjects
    pt = subjects{sind};
    load(fullfile(extractedpath,strcat(pt,'_',task,'.mat')));
    ROIs = load(fullfile(extractedpath,[pt,'_ROIs.mat']));
    % Combine multiple runs/sessions if needed
    if strcmp(pt,'P21N004')
        % For P21N004, use Reverse context data from run 2 and
        % Canonical data from run 3 (patient had multiple sessions;
        % see README file in the patient data folder)
        infoNew.chans = info(2).chans;
        temp1 = info(2).expt(651:end,:);
        temp2 = info(3).expt;
        infoNew.expt = [temp2;temp1];
    else
        infoNew = info;
    end
    if strcmp(pt,'P21N001') % patient with EEG and sEEG data
        nsEEG = numel(infoNew.chans)-3; % don't use the 3 EEG channels at the end
    else % patients with only sEEG data
        nsEEG = numel(infoNew.chans);
    end
    load(fullfile(extractedpath,['preprocessed_',pt,'_',task]));
    ntrials = size(env_HGA_sEEG,3);
    nsamps = size(env_HGA_sEEG,1);
    if strcmp(pt,'P21N001')
        clear rois
        for k = 1:nsEEG
            rois{k,:} = ROIs.anat.chans(k).roi;
        end
    else
        rois = ROIs.roi;
    end
    
    % Perform baseline correction for sEEG response after taking log 
    env_HGA_sEEG(env_HGA_sEEG < eps) = eps; % Replace anything < eps by eps
    env_HGA_sEEG = log10(env_HGA_sEEG);
    bline = 1:(-epochWin_sec(1)*fs_EEG); % samples in baseline period
    bline_mu_neuralResp = mean(env_HGA_sEEG(bline,:,:));
    bline_std_neuralResp = std(env_HGA_sEEG(bline,:,:));
    env_HGA_sEEG = (env_HGA_sEEG - bline_mu_neuralResp)./bline_std_neuralResp; % z-score
    
    % Drop sEEG trials and channels having: neural response^2 > 4 STD
    % above the mean across epochs or < 4 STD below the mean across epochs,
    % by replacing them with NaN.
    x = squeeze(mean(mean(env_HGA_sEEG.^2)));
    threshUpper = median(x)+4*mad(x,1);
    threshLower = median(x)-4*mad(x,1);
    bads = ((x > threshUpper) | (x < threshLower));
    env_HGA_sEEG(:,:,bads) = nan;
    
    % Select only channels that respond to the stimuli
    respWin_sec = [0,300]/1000; % response window relative to stim onset (in seconds)
    respWin_samp = round(fs_EEG*[-epochWin_sec(1)+1/fs_EEG+respWin_sec(1),...
        -epochWin_sec(1)+1/fs_EEG+respWin_sec(2)]);
    respWin_inds = respWin_samp(1):respWin_samp(2);
    aveFactor = 1/sqrt(size(env_HGA_sEEG,3)-sum(bads));
    evokedResp = nanmean(env_HGA_sEEG,3);
    chansWithResponse = (max(evokedResp(respWin_inds,:)) > 1.96*aveFactor) | ...
        (mean(evokedResp(respWin_inds,:)) > 1*aveFactor);
    indChansWithResp = find(chansWithResponse);
    nChansWithResp = numel(indChansWithResp);
    
    resp_window = (-epochWin_sec(1)*fs_EEG+1):nsamps;
    dat = [];
    regs = {};
    for kr = 1:nChansWithResp
        ch = indChansWithResp(kr);
        roiChanWithResp = rois(ch);
        for ks = 1:numel(subsetRegions)
            ind_temp = strfind(roiChanWithResp,subsetRegions(ks));
            ind_temp = find(~cellfun(@isempty,ind_temp));
            if ~isempty(ind_temp)
                dat = cat(2,dat,env_HGA_sEEG(resp_window,ch,:));
                regs = [regs,roiChanWithResp];
            end
        end
    end
    allDat{sind}.dat = dat;
    allDat{sind}.regs = regs;
    allDat{sind}.beeOrPea = infoNew.expt.resp;
    allDat{sind}.F0 = infoNew.expt.F0;
    allDat{sind}.VOT = infoNew.expt.VOT;
    allDat{sind}.isTest = infoNew.expt.isTest;
end
    
save('datForClassifier.mat','allDat');

