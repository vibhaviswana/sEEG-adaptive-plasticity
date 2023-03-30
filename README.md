# sEEG-adaptive-plasticity

This repository contains custom code used for a project on the analysis of neural signatures of adaptive plasticity in speech perception with intracranial stereo-electroencephalography (sEEG).

Copyright 2021-23 Vibha Viswanathan. All rights reserved.

## Basic usage

The code should be run using the following steps: 

- Run ```preprocessing.m``` after changing patient ID in the code as needed. This script extracts sEEG data from raw .nsx files, computes high-gamma power, and saves pre-processed data to a .mat file.
- To run the classifier analysis, run ```percWts_prepForClassifier.m```, followed by ```percWts_CanonicalClassifier.py```, ```percWts_ReverseClassifier.py```, and ```percWts_compareClassifiers.py```.
- To run the functional connectivity analysis, run ```percWts_functionalConnectivity.m```.


> **_NOTE:_** The functional connectivity code calls the function ```mtcoh.m```, which is available as part of the [neuroutils toolbox](https://github.com/vibhaviswana/neuroutils).


