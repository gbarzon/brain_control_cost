import numpy as np
from mne.io import read_raw_eeglab
from mne import events_from_annotations
from scipy.ndimage import gaussian_filter1d
from string import ascii_uppercase

def smoothing(data, sigma):
    if sigma>0:
        return gaussian_filter1d(data, sigma, axis=1)
    else:
        return data

def back_fitting(data, maps):
    activation = maps.dot(data)
    return np.argmax(np.abs(activation), axis=0)

def load_dataset(subj, sfreq=250, lfreq=1, hfreq=40):
    subj = str(subj)

    # Add 0 if subj smaller than 10
    if len(subj)<2:
        subj = '0'+subj

    folder = '/data/barzon/stroop/'
    name = folder + f'ss{subj}_clean.set'

    # Read dataset
    data = read_raw_eeglab(name, preload=True)

    if sfreq is not None:
        data = data.resample(sfreq)

    if lfreq is not None or hfreq is not None:
        data = data.filter(lfreq, hfreq)

    return data

def load_info(subj=1):
    return load_dataset(subj).info

def get_names(n_states):
    return list(ascii_uppercase)[:n_states]