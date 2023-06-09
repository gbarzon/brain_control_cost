import numpy as np

import mne

import utils.mne_microstates as microstates
from utils.utils import *

### Utils
def find_statistics_window(data, maps, n_states):
    # Backfitting
    tmp = back_fitting(data, maps)
    
    # Occurrences
    occurrences = mycounts(tmp, np.arange(n_states))
            
    # Transitions
    transitions = tmp_count_trans(tmp, n_states, lag=1)
    
    # Durations
    durations = find_duration(tmp, n_states)
            
    return occurrences, transitions, durations

def find_statistics(data, maps, tasks, events, events_id):
    n_states = len(maps)
    occurrences = np.zeros((len(tasks),n_states))
    transitions = np.zeros((len(tasks),n_states,n_states))
    durations = [[[] for _ in range(n_states)] for _ in range(len(tasks))]

    for i_task, task in enumerate(tasks):
        #print(task)
        
        if task=='REST':
            rest = data[:,:events[0,0]]
            #print(rest.shape)
            occ, ttt, dur = find_statistics_window(rest, maps, n_states)
            occurrences[i_task] += occ
            transitions[i_task] += ttt
            for s in range(n_states):
                durations[i_task][s] += dur[s]
            
        else:
            # Find start and stop
            idx = np.where(events[:,2]==events_id[task])[0]
            start = events[idx][:,0]
            stop = events[idx+1][:,0]
    
            if len(start)!=len(stop):
                print('WARNING: length are different...')

            for i_event in range(len(start)):
                # Get task data
                tmp = data[:,start[i_event]:stop[i_event]]
                occ, ttt, dur = find_statistics_window(tmp, maps, n_states)
                occurrences[i_task] += occ
                transitions[i_task] += ttt
                #print(i_event, occurrences[i_task], transitions[i_task])
                
                for s in range(n_states):
                    durations[i_task][s] += dur[s]
            
    return occurrences, transitions, durations

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