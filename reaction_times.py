import numpy as np

from mne import events_from_annotations

from utils.utils import *

### Define variables
n_subj = 44
subjects = np.arange(n_subj) + 1

tasks = ['S 21', 'S 51', 'S 71', 'S 22', 'S 52', 'S 72']

### Compute average reaction times
def get_rt(events, events_id, fs=125):
    ress = np.zeros(len(tasks))

    for i, task in enumerate(tasks):
        print(task)
    
        # Find start and stop
        idx = np.where(events[:,2]==events_id[task])[0]
        start = events[idx][:,0]
        stop = events[idx+1][:,0]
    
        ress[i] = np.mean((stop-start))/fs
        
    return ress

### Loop over subjects
rt = np.zeros((len(tasks),n_subj))

for i_subj, subj in enumerate(subjects):
    print(f'[*] SUBJECT {subj}/{n_subj}')
    # Load dataset
    dataset = load_dataset(subj)
    
    # Get events
    events, events_id = events_from_annotations(dataset)
    
    # Compute RT
    #tmp_rt = get_rt(events, events_id)
    #rt.append(tmp_rt)
    rt[:,i_subj] = get_rt(events, events_id)
    
### Store results
np.save('results/rt.npy', rt.T)