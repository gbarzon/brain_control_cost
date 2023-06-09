import numpy as np
from mne import events_from_annotations

import utils.mne_microstates as microstates
from utils.utils import *

from scipy.signal import find_peaks

### Load infos
info = load_info()
n_subj = 44
subjects = np.arange(n_subj) + 1

### Set params for clustering
min_peak_dist = 5
sigma = 5
n_states = np.arange(2,15)

### Concatenate GFP peaks
data_gfp = []

for subj in subjects:
    print(f'############ SUBJ {subj} ############')
    print('Loading data...')
    dataset = load_dataset(subj)
    
    # Get events
    events, events_id = events_from_annotations(dataset)
    
    # Get resting state
    data = dataset.get_data()
    data = data[:,:events[0,0]]
    print(f'Resting data shape: {data.shape}')
    
    # Temporal smoothing
    data_smooth = smoothing(data, sigma=sigma)

    # Compute gfp
    gfp = np.std(data, axis=0)
    gfp_smooth = np.std(data_smooth, axis=0)

    # Find peaks
    peaks, _ = find_peaks(gfp_smooth, distance=min_peak_dist)
    n_peaks = len(peaks)
    
    data_gfp.append(data_smooth[:,peaks])
    
### Compute minimum number of peaks
min_peaks = np.inf
for tmp in data_gfp:
    if tmp.shape[1] < min_peaks:
        min_peaks = tmp.shape[1]
        
new_res = []
for tmp in data_gfp:
    new_res.append(tmp[:,:min_peaks])
    
new_res_conc = np.concatenate(new_res, axis=1)

### Run clustering
for i, tmp in enumerate(n_states):
    tmp_res = microstates.segment_peaks(new_res_conc*1e6, n_states=tmp, n_inits=10, max_iter=1000, thresh=1e-6,
            normalize=False, min_peak_dist=min_peak_dist, max_n_peaks=None,
            return_polarity=False, random_state=None, verbose=None)
    res_validation.append(tmp_res)
    
return res_validation