# FONet

## Introduction:
This page contains the necessary code to run FONet, a deep learning algorithm for automatically detecting interictal epileptiform discharges (IEDs) on Foramen Ovale EEG (FO-EEG) recordings. The methods used to develop this algorithm and the expected performance are presented in [https://doi.org/10.1093/sleep/zsaa112.](https://doi.org/10.1016%2Fj.clinph.2019.09.031)

## How to use:
Step 1: Import the model fo_spike_detector
```
from fonet import fo_spike_detector
```

Step 2: Construct an instance of the fo_spike_detector, and load its weights (weights are included in this repo)
```
model = fo_spike_detector(dropout_rate=0)
model.load_weights('FOnet.h5')
```

Step 3: Predict presence of IEDs using the method .predict() and input **x**
```
y_hat = model.predict(x)
```

**x** is a sequence of EEG epochs consisting of a single bipolar FO-EEG channel.

The output **yhat** is a parallel array of **x** consisting of numbers between 0 and 1, which represent the probability that the corresponding EEG epoch contains an IED.

**Note**: If your EEG epochs have multiple channels, then each channel is individually provided to FONet, and the outputs for each channel are averaged.

An example is provided in **apply_fonet.ipynb**, where FOnet is applied on a single channel bipolar FO-EEG recording.

## Environment:
Python 3.6.9

Tensorflow 2.5.1

## Citation:

Please cite the following paper [https://doi.org/10.1093/sleep/zsaa112.](https://doi.org/10.1016%2Fj.clinph.2019.09.031) when using this algorithm. 
