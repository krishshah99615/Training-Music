import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from math import pi
from scipy.io import wavfile


def create_sound(s,MAX_DURATION,MIN_VAL, MAX_VAL,SAMPLING_RATE):

    #Load the training data
    df = pd.read_csv('model_log.csv')
    l = df[s].values

    # Map values into frquency range
    m = interp1d([min(l), max(l)], [MIN_VAL, MAX_VAL])
    list_of_f = [m(x) for x in l]

    #empty list for storing cos wave
    x = []

    #list of values for each frquency
    t = np.arange(0, MAX_DURATION, 1. / SAMPLING_RATE)

    for f in list_of_f:
        cos_wav = 10000 * np.cos(2 * pi * f * t)  # generated signals
        x  = np.hstack((x, cos_wav))

    wavfile.write(f'{s}.wav', SAMPLING_RATE, x.astype(np.dtype('i2')))

for s in ['loss','val_loss','accuracy','val_accuracy']:
    create_sound(s,0.15,1000,3000,22050)

