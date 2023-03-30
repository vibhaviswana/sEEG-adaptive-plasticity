% Preprocessing script to extract high-gamma power (HGA) from raw sEEG
% data.
%
% Copyright 2021-23 Vibha Viswanathan. All rights reserved.

pt = 'P21N008'; % patient ID
task = 'percWts'; % task to extract
datarootpath = '/OneDrive/1. Recording & Stimulation of iEEG';
addpath(genpath(fullfile(datarootpath,'code')));
extractedpath = fullfile('./Extracted'); % location to save pre-processed data
[dataCont,info] = extractRippleData(pt,task,datarootpath);
fs_EEG = 1000; % EEG sampling rate in Hz
epochWin = [-100 stimDur+200]; % epoch window relative to stimulus onset in samples (or ms)
tLfp = epochWin(1):epochWin(2);

if strcmp(pt,'P21N001')
    inds_sEEG = 1:226; % channel indices for sEEG
elseif strcmp(pt,'P21N004')
    dataCont = dataCont(2:3); 
    info = info(2:3);
    inds_sEEG = 1:size(dataCont{1},2);
else
    inds_sEEG = 1:size(dataCont{1},2);
end

nSess = length(dataCont); % # of sessions over which data were collected

% Parameters for high-gamma-band filtering
HGrange = [75, 150]; % high-gamma range (in Hz) used by Mesgarani and Chang (2012)
% Transition band = 15 Hz. Filter order = 1/transition band = 68 samples (or ms)
ord = 68;
b_HG = fir1(ord, HGrange/(fs_EEG/2));  % linear phase filter (constant group delay)

% Parameters for HGA envelope extraction via low-pass filtering
% HGrange(2)-HGrange(1) = 75, so go up to 60 Hz for cutoff.
HGAenvCutoff = 60; % Hz
% Define non-negative filter below
ord = ceil(fs_EEG/HGAenvCutoff);
b_HGenv = hanning(ord);

for iSess = 1:nSess
    thisDat = dataCont{iSess};
    % Common average referencing
    cmavg_sEEG = mean(thisDat(:,inds_sEEG),2); 
    datCar = thisDat(:,inds_sEEG) - cmavg_sEEG; 
    % Filter in high-gamma range using filtfilt to correct for group delay
    datCarHGA = filtfilt(b_HG,1,datCar); 
    % Extract envelope in high-gamma band by half-wave rectification
    % followed by low-pass filtering 
    datCarHGA(datCarHGA<0) = 0;
    datCarHGA = filtfilt(b_HGenv,1,datCarHGA);
    % Get onset index
    [~,iOns] = arrayfun(@(x) min(abs(info(iSess).t - x)),info(iSess).expt.tStimOns);
    % Epoch the data
    datTmpHGA = arrayfun(@(x) datCarHGA(x + tLfp,:),iOns,'UniformOutput',false);
    datHGA{iSess} = cat(3,datTmpHGA{:}); 
end

if strcmp(pt,'P21N004')
    % For P21N004, use Reverse context data from run 2 and
    % Canonical data from run 3 (patient had multiple sessions;
    % see README file in the patient data folder)
    temp1 = datHGA{1};
    temp1 = temp1(:,:,651:end); % Reverse context from run 2
    temp2 = datHGA{2}; % Canonical context from run 3
    env_HGA_sEEG = cat(3,temp2,temp1); % put Canonical before Reverse
else
    env_HGA_sEEG = cat(3,datHGA{:}); 
end

save(fullfile(extractedpath,['preprocessed_',pt,'_',task,'.mat']),...
    'pt','fs_EEG','env_HGA_sEEG','epochWin','-v7.3');

