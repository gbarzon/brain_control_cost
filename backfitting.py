import numpy as np

from mne import events_from_annotations

from utils.utils import *

### Load infos
info = load_info()

n_subj = 44
subjects = np.arange(n_subj) + 1

tasks = ['REST', 'S 21', 'S 51', 'S 71', 'S 22', 'S 52', 'S 72']

sigma = 5
min_peak_dist = 5
n_states = 9

maps = np.load('results/best_maps.npy')

### Perform backfitting
occurrences = np.zeros((n_subj,len(tasks),n_states))
transitions = np.zeros((n_subj,len(tasks),n_states,n_states))
durations = []

to_reject = []

for i_subj, subj in enumerate(subjects):
    print(f'[*] SUBJECT {subj}/{n_subj}')
    # Load dataset
    dataset = load_dataset(subj)
    
    # Get data
    data = dataset.get_data()
    
    # Get events
    events, events_id = events_from_annotations(dataset)
    
    # Smoothing data
    data = smoothing(data, sigma)
    
    # Check if all microstates appears
    tmp = back_fitting(data, maps)
    ss = np.unique(tmp)
    if len(ss)!=n_states:
        print('WARNING: not all microstates are present...')
        to_reject.append(subj)
    
    # Find statistics
    occurrences[i_subj], transitions[i_subj], dur = find_statistics(data, maps, tasks, events, events_id)
    durations.append(dur)
    
### Store results
np.save('results/occurrence.npy', occurrences)
np.save('results/transitions.npy', transitions)