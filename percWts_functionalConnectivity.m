%% Script to run functional connectivity analysis on pre-processed sEEG high-gamma-band power.
%
% Copyright 2021-23 Vibha Viswanathan. All rights reserved.

fmax = 30; % in Hz
fs = 1000; % in Hz
nsamps = 551;
subjects = {'P21N001','P21N004','P21N007','P21N008','P22N007'}; 
nsubs = numel(subjects);
datarootpath = '/OneDrive/1. Recording & Stimulation of iEEG';
addpath(genpath(fullfile(datarootpath,'code')));
extractedpath = fullfile('./Extracted'); % location of pre-processed data
task = 'percWts'; % task to extract
auditoryCortex = {'(HESCHL) R','(HESCHL) L','(STG) R','(STG) L',...
    '(STS) R','(STS) L',...
    'Temporal-to-Parietal (GapMap) R','Temporal-to-Parietal (GapMap) L'};
parietalCortex = {'(IPL) R','(IPL) L','(IPS) R', '(IPS) L',...
    '(SPL) R', '(SPL) L', ...
    '(PostCG) R','(PostCG) L','(PostCS) L'};
frontalCortex = {'(IFG) R','(IFG) L','(OFC) R','(OFC) L',...
    'Frontal-I (GapMap) R','Frontal-I (GapMap) L',...
    'Frontal-II (GapMap) R','Frontal-II (GapMap) L',...
    '(SFS) R','(SFS) L',...
    '(POperc) R','(POperc) L',...
    '(Frontal Operculum) R','(Frontal Operculum) L',...
    '(PreCG) R','(PreCG) L','(preSMA, mesial SFG) R','(SMA, mesial SFG) L'};
otherCortex = {'(Insula) R','(Insula) L','(sACC) R'};
subCortical = {'(Hippocampus) R','(Hippocampus) L','(Amygdala) R'};
regions = [auditoryCortex,parietalCortex,frontalCortex,otherCortex,subCortical];
nregions = numel(regions);
subsetRegions = {'(HESCHL) L','(STS) L','Temporal-to-Parietal (GapMap) L','(HESCHL) R','(STG) R','(STG) L',...
    '(OFC) R','(STS) R','Temporal-to-Parietal (GapMap) R','(IPL) R','(POperc) R','(Insula) R',...
    '(IPL) L','(PostCG) R','(Insula) L','Frontal-II (GapMap) L','Frontal-II (GapMap) R',...
    '(IFG) R','(SFS) L','(Amygdala) R','(PostCG) L','(IFG) L','(PostCS) L',...
    '(IPS) R','(IPS) L','(SPL) L','(PreCG) R'}; % non-visual stimulus responsive regions (based on latency analysis)
nSubsetregs = numel(subsetRegions);

whichregs = zeros(nSubsetregs,1);
for k = 1:nSubsetregs
    whichregs(k) = find(ismember(regions,subsetRegions{k}));
end

stimDur = 250; % in ms
epochWin_sec = [-100 stimDur+200]/1000; % epoch window relative to stim onset (in seconds)
ntrialcanpassive = 600; % number of Canonical passive stimuli per subject
ntrialcantest = 50; % number of Canonical Test stimuli per subject
nw = 1; % frequency resolution = 2nw/time-duration = 2/(0.5 s) = 4 Hz.
ntapers = 2*nw-1;
nfft = 2^nextpow2(nsamps);
freqs = (0:(nfft-1))*fs/nfft;
nfreqs = sum(freqs<=fmax);
ncoherences = nSubsetregs*(nSubsetregs-1)/2;

regionindices = zeros(ncoherences,2);
k = 0;
for m1 = 1:nSubsetregs
    for m2 = (m1+1):nSubsetregs
        k = k+1;
        regionindices(k,:) = [m1,m2];
    end
end

coh_allsubs_can_passive = zeros(nsubs,ncoherences,nfreqs);
coh_allsubs_rev_passive = zeros(nsubs,ncoherences,nfreqs);
coh_allsubs_can_test = zeros(nsubs,ncoherences,nfreqs);
coh_allsubs_rev_test = zeros(nsubs,ncoherences,nfreqs);
niters = 100;
null_allsubs_diff_passive = zeros(nsubs,niters,ncoherences,nfreqs);
null_allsubs_diff_test = zeros(nsubs,niters,ncoherences,nfreqs);

for sind = 1:nsubs % iterate over subjects
    pt = subjects{sind};
    load(fullfile(extractedpath,strcat(pt,'_',task,'.mat')));
    load(fullfile(extractedpath,['preprocessed_',pt,'_',task]));
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
    passiveinds = (infoNew.expt.isTest==false);
    caninds = (strcmp(infoNew.expt.condition,'Canonical'));
    revinds = (strcmp(infoNew.expt.condition,'Reverse'));
    can_passiveinds = find(passiveinds&caninds);
    rev_passiveinds = find(passiveinds&revinds);
    testinds = (infoNew.expt.isTest==true);
    can_testinds = find(testinds&caninds);
    rev_testinds = find(testinds&revinds);
    if strcmp(pt,'P21N001') % patient with EEG and sEEG data
        nsEEG = numel(infoNew.chans)-3; % don't use the 3 EEG channels at the end
    else % patients with only sEEG data
        nsEEG = numel(infoNew.chans);
    end
    if strcmp(pt,'P21N001')
        clear rois
        for k = 1:nsEEG
            rois{k,:} = ROIs.anat.chans(k).roi;
        end
    else
        rois = ROIs.roi;
    end

    % Perform baseline correction for sEEG neural response after taking log
    env_HGA_sEEG(env_HGA_sEEG < eps) = eps; % Replace anything < eps by eps
    env_HGA_sEEG = log10(env_HGA_sEEG);
    bline = 1:(-epochWin(1)); % samples in baseline period
    bline_mu_envHGAsEEG = mean(env_HGA_sEEG(bline,:,:));
    bline_std_envHGAsEEG = std(env_HGA_sEEG(bline,:,:));
    env_HGA_sEEG = (env_HGA_sEEG - bline_mu_envHGAsEEG)./bline_std_envHGAsEEG; % z-score

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
    respWin_samp = round(fs*[-epochWin_sec(1)+1/fs+respWin_sec(1),...
        -epochWin_sec(1)+1/fs+respWin_sec(2)]);
    respWin_inds = respWin_samp(1):respWin_samp(2);
    aveFactor = 1/sqrt(size(env_HGA_sEEG,3)-sum(bads));
    evokedResp = nanmean(env_HGA_sEEG,3);
    chansWithResponse = ((max(evokedResp(respWin_inds,:)) > 1.96*aveFactor) | ...
        (mean(evokedResp(respWin_inds,:)) > 1*aveFactor)) & ...
        (min(evokedResp(respWin_inds,:)) > -0.2);
    indChansWithResp = find(chansWithResponse);

    can_thissub_allregs_passive = nan(nSubsetregs,nsamps,ntrialcanpassive);
    rev_thissub_allregs_passive = nan(nSubsetregs,nsamps,ntrialcanpassive);
    can_thissub_allregs_test = nan(nSubsetregs,nsamps,ntrialcantest);
    rev_thissub_allregs_test = nan(nSubsetregs,nsamps,ntrialcantest);
    can_thissub_allregs_combined = nan(nSubsetregs,nsamps,ntrialcanpassive+ntrialcantest);
    rev_thissub_allregs_combined = nan(nSubsetregs,nsamps,ntrialcanpassive+ntrialcantest);

    for m = 1:nSubsetregs % iterate over brain regions of interest
        creg = whichregs(m);
        ind_reg = strfind(rois,regions{creg});
        ind_reg = (find(~cellfun(@isempty,ind_reg)))';
        nrespchans = 0;
        for k = 1:numel(ind_reg)
            ch = ind_reg(k);
            if find(indChansWithResp==ch) % ch is responsive to our stimuli
                nrespchans = nrespchans + 1;
            end
        end
        can_reg_passive = nan(nrespchans,nsamps,ntrialcanpassive);
        rev_reg_passive = nan(nrespchans,nsamps,ntrialcanpassive);
        can_reg_test = nan(nrespchans,nsamps,ntrialcantest);
        rev_reg_test = nan(nrespchans,nsamps,ntrialcantest);
        cnt = 0;
        for k = 1:numel(ind_reg)
            ch = ind_reg(k);
            if find(indChansWithResp==ch) % ch is responsive to our stimuli
                cnt = cnt + 1;
                tempHGA = squeeze(env_HGA_sEEG(:,ch,:));
                can_reg_passive(cnt,:,:) = tempHGA(:,can_passiveinds);
                rev_reg_passive(cnt,:,:) = tempHGA(:,rev_passiveinds);
                can_reg_test(cnt,:,:) = tempHGA(:,can_testinds);
                rev_reg_test(cnt,:,:) = tempHGA(:,rev_testinds);
            end
        end
        if (nrespchans~=0)
            can_thissub_allregs_passive(m,:,:) = squeeze(nanmean(can_reg_passive,1));
            rev_thissub_allregs_passive(m,:,:) = squeeze(nanmean(rev_reg_passive,1));
            can_thissub_allregs_test(m,:,:) = squeeze(nanmean(can_reg_test,1));
            rev_thissub_allregs_test(m,:,:) = squeeze(nanmean(rev_reg_test,1));
        elseif (nrespchans==0)
            can_thissub_allregs_passive(m,:,:) = -inf;
            rev_thissub_allregs_passive(m,:,:) = -inf;
            can_thissub_allregs_test(m,:,:) = -inf;
            rev_thissub_allregs_test(m,:,:) = -inf;
        end
    end

    % remove NaN trials (noisy neuralResp) -- passive
    inds = squeeze(sum(squeeze(sum(isnan(can_thissub_allregs_passive),1)),1));
    can_thissub_allregs_passive_new = can_thissub_allregs_passive(:,:,~inds);
    inds = squeeze(sum(squeeze(sum(isnan(rev_thissub_allregs_passive),1)),1));
    rev_thissub_allregs_passive_new = rev_thissub_allregs_passive(:,:,~inds);
    nt1 = size(can_thissub_allregs_passive_new,3);
    nt2 = size(rev_thissub_allregs_passive_new,3);
    ntrialpassive2 = min(nt1,nt2); % determine # ntrials available per region
    can_thissub_allregs_passive_new = can_thissub_allregs_passive_new(:,:,1:ntrialpassive2);
    rev_thissub_allregs_passive_new = rev_thissub_allregs_passive_new(:,:,1:ntrialpassive2);
    % remove NaN trials (noisy neuralResp) -- test
    inds = squeeze(sum(squeeze(sum(isnan(can_thissub_allregs_test),1)),1));
    can_thissub_allregs_test_new = can_thissub_allregs_test(:,:,~inds);
    inds = squeeze(sum(squeeze(sum(isnan(rev_thissub_allregs_test),1)),1));
    rev_thissub_allregs_test_new = rev_thissub_allregs_test(:,:,~inds);
    nt1 = size(can_thissub_allregs_test_new,3);
    nt2 = size(rev_thissub_allregs_test_new,3);
    ntrialtest2 = min(nt1,nt2); % determine # ntrials available per region
    can_thissub_allregs_test_new = can_thissub_allregs_test_new(:,:,1:ntrialtest2);
    rev_thissub_allregs_test_new = rev_thissub_allregs_test_new(:,:,1:ntrialtest2);

    % Compute within-subject functional connectivity (PLV) between regions
    k = 0;
    for m1 = 1:nSubsetregs
        for m2 = (m1+1):nSubsetregs
            k = k+1;
            creg1 = whichregs(m1);
            creg2 = whichregs(m2);
            x = squeeze(can_thissub_allregs_passive_new(m1,:,:))';
            y = squeeze(can_thissub_allregs_passive_new(m2,:,:))';
            if (any(isinf(x(:))) || any(isinf(y(:)))) % either creg1 or creg2 was not measured
                coh_allsubs_can_passive(sind,k,:) = nan;
                coh_allsubs_rev_passive(sind,k,:) = nan;
                coh_allsubs_can_test(sind,k,:) = nan;
                coh_allsubs_rev_test(sind,k,:) = nan;
            else
                % Passive
                x = x - mean(x,2); % demean
                y = y - mean(y,2); % demean
                [coh_allsubs_can_passive(sind,k,:),freqs] = mtcoh(x, y, nw, fs, 'True', fmax);
                x = squeeze(rev_thissub_allregs_passive_new(m1,:,:))';
                x = x - mean(x,2); % demean
                y = squeeze(rev_thissub_allregs_passive_new(m2,:,:))';
                y = y - mean(y,2); % demean
                [coh_allsubs_rev_passive(sind,k,:),freqs] = mtcoh(x, y, nw, fs, 'True', fmax);
                % Test
                x = squeeze(can_thissub_allregs_test_new(m1,:,:))';
                x = x - mean(x,2); % demean
                y = squeeze(can_thissub_allregs_test_new(m2,:,:))';
                y = y - mean(y,2); % demean
                [coh_allsubs_can_test(sind,k,:),freqs] = mtcoh(x, y, nw, fs, 'True', fmax);
                x = squeeze(rev_thissub_allregs_test_new(m1,:,:))';
                x = x - mean(x,2); % demean
                y = squeeze(rev_thissub_allregs_test_new(m2,:,:))';
                y = y - mean(y,2); % demean
                [coh_allsubs_rev_test(sind,k,:),freqs] = mtcoh(x, y, nw, fs, 'True', fmax);
            end
        end
    end
    % Within-subject null distribution for PLV difference
    % Nonparametric permutation-based approach
    for m = 1:niters
        k = 0;
        p_passive = logical(binornd(1,0.5,[1,ntrialpassive2]));
        p_test = logical(binornd(1,0.5,[1,ntrialtest2]));
        for m1 = 1:nSubsetregs
            for m2 = (m1+1):nSubsetregs
                k = k+1;
                creg1 = whichregs(m1);
                creg2 = whichregs(m2);
                % Check if regions were measured in subject
                x = squeeze(can_thissub_allregs_passive_new(m1,:,:))';
                y = squeeze(can_thissub_allregs_passive_new(m2,:,:))';
                if (any(isinf(x(:))) || any(isinf(y(:)))) % either creg1 or creg2 was not measured
                    null_allsubs_diff_passive(sind,m,k,:) = nan;
                    null_allsubs_diff_test(sind,m,k,:) = nan;
                else
                    % Null for "Reverse - Canonical" difference: Passive
                    mix1reg1 = [squeeze(can_thissub_allregs_passive_new(m1,:,p_passive)),...
                        squeeze(rev_thissub_allregs_passive_new(m1,:,~p_passive))]';
                    mix2reg1 = [squeeze(rev_thissub_allregs_passive_new(m1,:,p_passive)),...
                        squeeze(can_thissub_allregs_passive_new(m1,:,~p_passive))]';
                    mix1reg2 = [squeeze(can_thissub_allregs_passive_new(m2,:,p_passive)),...
                        squeeze(rev_thissub_allregs_passive_new(m2,:,~p_passive))]';
                    mix2reg2 = [squeeze(rev_thissub_allregs_passive_new(m2,:,p_passive)),...
                        squeeze(can_thissub_allregs_passive_new(m2,:,~p_passive))]';
                    mix1reg1 = mix1reg1 - mean(mix1reg1,2);
                    mix1reg2 = mix1reg2 - mean(mix1reg2,2);
                    mix2reg1 = mix2reg1 - mean(mix2reg1,2);
                    mix2reg2 = mix2reg2 - mean(mix2reg2,2);
                    [coh_mix1,freqs] = mtcoh(mix1reg1, mix1reg2, nw, fs, 'True', fmax);
                    [coh_mix2,freqs] = mtcoh(mix2reg1, mix2reg2, nw, fs, 'True', fmax);
                    null_allsubs_diff_passive(sind,m,k,:) = coh_mix2 - coh_mix1;
                    % Null for "Reverse - Canonical" difference: Test
                    mix1reg1 = [squeeze(can_thissub_allregs_test_new(m1,:,p_test)),...
                        squeeze(rev_thissub_allregs_test_new(m1,:,~p_test))]';
                    mix2reg1 = [squeeze(rev_thissub_allregs_test_new(m1,:,p_test)),...
                        squeeze(can_thissub_allregs_test_new(m1,:,~p_test))]';
                    mix1reg2 = [squeeze(can_thissub_allregs_test_new(m2,:,p_test)),...
                        squeeze(rev_thissub_allregs_test_new(m2,:,~p_test))]';
                    mix2reg2 = [squeeze(rev_thissub_allregs_test_new(m2,:,p_test)),...
                        squeeze(can_thissub_allregs_test_new(m2,:,~p_test))]';
                    mix1reg1 = mix1reg1 - mean(mix1reg1,2);
                    mix1reg2 = mix1reg2 - mean(mix1reg2,2);
                    mix2reg1 = mix2reg1 - mean(mix2reg1,2);
                    mix2reg2 = mix2reg2 - mean(mix2reg2,2);
                    [coh_mix1,freqs] = mtcoh(mix1reg1, mix1reg2, nw, fs, 'True', fmax);
                    [coh_mix2,freqs] = mtcoh(mix2reg1, mix2reg2, nw, fs, 'True', fmax);
                    null_allsubs_diff_test(sind,m,k,:) = coh_mix2 - coh_mix1;
                end
            end
        end
    end
end

% Average functional connectivity and null distribution over subjects
coh_can_passive = squeeze(nanmean(coh_allsubs_can_passive,1));
coh_rev_passive = squeeze(nanmean(coh_allsubs_rev_passive,1));
coh_can_test = squeeze(nanmean(coh_allsubs_can_test,1));
coh_rev_test = squeeze(nanmean(coh_allsubs_rev_test,1));

coh_diff_passive = coh_rev_passive - coh_can_passive;
coh_diff_test = coh_rev_test - coh_can_test;

null_diff_passive = squeeze(nanmean(null_allsubs_diff_passive,1));
null_diff_test = squeeze(nanmean(null_allsubs_diff_test,1));


%% Average PLV within the delta-theta band

bandstrs = {'Delta-Theta'};
nbands = numel(bandstrs);
bandnos = 1:nbands;
deltathetarange = [0, 7];
DTinds = (freqs >= deltathetarange(1)) & (freqs <= deltathetarange(2));
bandCoh_diff_passive = zeros(nbands,ncoherences);
bandCoh_diff_test = zeros(nbands,ncoherences);
bandNull_diff_passive = zeros(niters,nbands,ncoherences);
bandNull_diff_test = zeros(niters,nbands,ncoherences);
for q = 1:nbands
    if (q==1)
        bandinds = DTinds;
    end
    bandCoh_diff_passive(q,:) = mean(coh_diff_passive(:,bandinds),2);
    bandCoh_diff_test(q,:) = mean(coh_diff_test(:,bandinds),2);
    bandNull_diff_passive(:,q,:) = mean(null_diff_passive(:,:,bandinds),3);
    bandNull_diff_test(:,q,:) = mean(null_diff_test(:,:,bandinds),3);
end


%% Compute pvals and threshold

mean_nullDiff_passive = reshape(mean(bandNull_diff_passive,1),[nbands,ncoherences]);
std_nullDiff_passive = reshape(std(bandNull_diff_passive,[],1),[nbands,ncoherences]);
mean_nullDiff_test = reshape(mean(bandNull_diff_test,1),[nbands,ncoherences]);
std_nullDiff_test = reshape(std(bandNull_diff_test,[],1),[nbands,ncoherences]);

pvals_diff_passive = zeros(size(bandCoh_diff_passive));
pvals_diff_test = zeros(size(bandCoh_diff_test));

for q = 1:nbands
    for r = 1:ncoherences
        % Passive
        if (bandCoh_diff_passive(q,r) >= 0)
            pvals_diff_passive(q,r) = 1-normcdf(bandCoh_diff_passive(q,r),...
                mean_nullDiff_passive(q,r),std_nullDiff_passive(q,r));
        else
            pvals_diff_passive(q,r) = normcdf(bandCoh_diff_passive(q,r),...
                mean_nullDiff_passive(q,r),std_nullDiff_passive(q,r));
        end
        % Test
        if (bandCoh_diff_test(q,r) >= 0)
            pvals_diff_test(q,r) = 1-normcdf(bandCoh_diff_test(q,r),...
                mean_nullDiff_test(q,r),std_nullDiff_test(q,r));
        else
            pvals_diff_test(q,r) = normcdf(bandCoh_diff_test(q,r),...
                mean_nullDiff_test(q,r),std_nullDiff_test(q,r));
        end
    end
end

% Account for the fact that 25% of the coherences across different pairs
% of regions are NaN, meaning that there was not even one subject who had
% simultaneous measurements from both regions in that pair. Discount these 
% when considering no. of coherences for multiple comparisons correction.
numNaNcoherences = sum(isnan(pvals_diff_passive));
nCoherencesEffective = ncoherences-numNaNcoherences;

FDR = 0.05;
% Passive
pvals_diff_sorted = (sort(pvals_diff_passive(:),'ascend'))';
fdr_line = (1:nCoherencesEffective)*FDR/nCoherencesEffective;
ind1 = (pvals_diff_sorted(~isnan(pvals_diff_sorted)) <= fdr_line);
if (sum(ind1) == numel(ind1))
    p_thresh = pvals_diff_sorted(sum(ind1)-1);
else
    p_thresh = pvals_diff_sorted(sum(ind1));
end
pos_fdr = find(pvals_diff_passive <= p_thresh); % positions meeting FDR threshold
bandCoh_diff_passive2 = zeros(size(bandCoh_diff_passive));
bandCoh_diff_passive2(pos_fdr) = bandCoh_diff_passive(pos_fdr);
pvals_diff_passive2 = pvals_diff_passive;
pvals_diff_passive2(pvals_diff_passive > p_thresh) = inf;
   
% Repopulate coherence and pvalue arrays with NaN for those region pairs
% that were not simultaneously measured in any subject. These will be
% plotted in gray
bandCoh_diff_passive2(isnan(bandCoh_diff_passive)) = nan;
pvals_diff_passive2(isnan(pvals_diff_passive)) = nan;


%% Plotting

cmap = cbrewer2('PrGn', 256);
cmap = [0.75*[1,1,1]; cmap; [0,0,0]]; % add grey color as first row (for NaN) and black as last row (for symmetric half of FC matrix)
regpairs = cell(ncoherences,1);
for k = 1:ncoherences
    r1 = regionindices(k,1);
    r2 = regionindices(k,2);
    regpairs{k} = [subsetRegions{r1}, ' and ', subsetRegions{r2}];
end

% Functional connectivity difference matrix
figure;
FCdiffmat_passive = zeros(nSubsetregs,nSubsetregs);
for k = 1:ncoherences
    r1 = regionindices(k,1);
    r2 = regionindices(k,2);
    FCdiffmat_passive(r1,r2) = bandCoh_diff_passive2(k);
    FCdiffmat_passive(r2,r1) = inf; % plot this in black
    FCdiffmat_passive(r2,r2) = inf; % plot this in black
    FCdiffmat_passive(r1,r1) = inf; % plot this in black
end
FCdiffmat_passive = FCdiffmat_passive';
FCdiffmat_passive(isnan(FCdiffmat_passive)) = -inf; % plot NaN values in gray by making them -Inf
h1 = imagesc(1:nSubsetregs,1:nSubsetregs,FCdiffmat_passive);
xticks(1:nSubsetregs);
yticks(1:nSubsetregs);
xticklabels(subsetRegions);
yticklabels(subsetRegions);
set(gca,'fontsize',14);
title('Passive');
grid on;
colormap(cmap);
limtemp = FCdiffmat_passive(~isinf(FCdiffmat_passive));
reallim = max(abs(limtemp(:)));
setlim = reallim + 0.01*reallim; % add buffer to ensure the max value doesn't become black and the min gray
clim([-setlim,setlim]);
ticks = reallim*(-1:0.5:1);
c = colorbar('fontsize',16);
c.Label.String = '"Reverse - Canonical" PLV difference';

% Example PLV spectrum difference
figure;
maxtemp = min(bandCoh_diff_passive2(~isnan(bandCoh_diff_passive2)));
indtemp = find(maxtemp==bandCoh_diff_passive2);
regpair_passive = indtemp;
plot(freqs(freqs<=fmax),coh_diff_passive(regpair_passive,:),'b','linewidth',2);
hold on;
nullmean = mean(squeeze(null_diff_passive(:,regpair_passive,:)),1);
nullstd = std(squeeze(null_diff_passive(:,regpair_passive,:)),[],1);
shadedErrorBar(freqs(freqs<=fmax),nullmean,nullstd,...
    'lineProps',{'k--','linewidth',2});
ylabel('"Reverse - Canonical" PLV spectrum difference','fontsize',18);
xlabel('Frequency (Hz)','fontsize',18);
legend({['Data from ',regpairs{regpair_passive}],'Noise floor'},...
    'location','southeast','fontsize',16);

% Brain network graph
figure;
whichpvals = pvals_diff_passive2;
whichbandcoh = bandCoh_diff_passive2;
fctag = 'Passive ';
edgeWts = (1./whichpvals)';
isPositivePLVdiff = (whichbandcoh>0);
g = graph(subsetRegions(regionindices(:,1)),subsetRegions(regionindices(:,2)),edgeWts);
g.Edges.isPositivePLVdiff = isPositivePLVdiff';
idx = find((g.Edges.Weight == 0) | isnan(g.Edges.Weight));
g = rmedge(g,idx);
idx = find(degree(g)==0);
g = rmnode(g,idx);
LWidths = log(g.Edges.Weight);
LWidths = LWidths - min(LWidths) + 4;
% Set different colors for negative and positive edge weights
endnodes = g.Edges.EndNodes(g.Edges.isPositivePLVdiff == 1,:);
p = plot(g,'Layout','circle','LineWidth',LWidths,...
    'MarkerSize',10,'NodeColor',0.5*ones(1,3),'EdgeColor',[175,141,195]/255);
highlight(p,endnodes(:,1),endnodes(:,2),'EdgeColor',[127,191,123]/255);
title('"Reverse - Canonical" brain network','fontsize',14);
xticks([]);
yticks([]);
% Fix tiny node labels issue
nl = p.NodeLabel;
p.NodeLabel = '';
xd = get(p, 'XData');
yd = get(p, 'YData');
mx = mean(xd);
my = mean(yd);
factor = 1.22;
xd2 = (xd - mx)*factor + mx;
yd2 = (yd - my)*factor + my;
text(xd2, yd2, nl, 'FontSize',20, 'HorizontalAlignment','center', 'VerticalAlignment','middle')


