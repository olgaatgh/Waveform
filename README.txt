README

Olga Dmitriyeva
waveform classification


The folder contains 3 files (including README.txt) and 2 folders

1)folder files_train contains 12 waveforms upon which the classifier is trained. The path to this folder needs to be assigned to the global variable MYPATH_TRAIN inside waveform_classification.py
2)folder files_predict should contain the waveforms against which the prediction will be made. The folder currently contains the same 12 waveforms. The path to this folder needs to be assigned to MYPATH_PREDICT global variable inside waveform_classification.py
3)waveform_classification.py - run to train classifier and make prediction. The script will generate the text file with file names and predicted labels: 1 - normal, 0 - anomalous
4)Dmitriyeva-analysis.pdf discusses the approach and implementation details
