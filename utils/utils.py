import numpy as np
from mne.io import read_raw_eeglab
from mne import events_from_annotations
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from string import ascii_uppercase
from random import sample
from itertools import groupby
from pandas import isnull

def smoothing(data, sigma):
    if sigma>0:
        return gaussian_filter1d(data, sigma, axis=1)
    else:
        return data

def back_fitting(data, maps):
    activation = maps.dot(data)
    return np.argmax(np.abs(activation), axis=0)

def load_dataset(subj, sfreq=125, lfreq=1, hfreq=40):
    subj = str(subj)
    
    # Add 0 if subj smaller than 10
    if len(subj)<2:
        subj = '0'+subj
        
    folder = 'data_500_1_100/'
    name = folder + f'ss{subj}_clean.set'
    
    return read_raw_eeglab(name, preload=True).resample(sfreq).filter(lfreq, hfreq)

def load_info(subj=1):
    return load_dataset(subj).info

def get_names(n_states):
    return list(ascii_uppercase)[:n_states]

def corr_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B).

    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)

def mycounts(arr1,arr2):
    return np.array([(arr1==i).sum() for i in arr2])

def find_duration(segmentation, n_states):
    ### Count consecutive occurrences (maybe adding 1 for beginning at half???)
    grouped = [[k, sum(1 for _ in g)] for k,g in groupby(segmentation)]
    grouped = np.array(grouped)
    
    durations = [[grouped[idx,1] for idx in np.where(grouped[:,0]==state)[0]] for state in range(n_states)]
    
    return durations

def tmp_count_trans(series, n_states, lag):
    q_matrix = np.zeros((n_states,n_states))
    
    for i in range(n_states):
        # init state
        s_in = series[:-lag] == i
    
        for j in range(n_states):
            # ending state
            s_out = series[lag:] == j
            # count occurrences
            q_matrix[i,j] = np.sum(s_in*s_out)
    
    return q_matrix

def count_trans(series, n_states, max_lag):
    q_matrix = np.zeros((max_lag,n_states,n_states))
    
    for i_lag, lag in enumerate(range(1,max_lag+1)):
        q_matrix[i_lag] = tmp_count_trans(series, n_states, lag)
    
    return q_matrix

def compute_statistics(data, maps, fs=125):
    n_states = len(maps)
    states = np.arange(n_states)
    
    ### Get segmentation
    segmentation = back_fitting(data, maps)
    
    ### Count consecutive occurrences (maybe adding 1 for beginning at half???)
    grouped = [[k, sum(1 for _ in g)] for k,g in groupby(segmentation)]
    grouped = np.array(grouped)
    # Discard first and last microstates
    #new_segmentation = segmentation[grouped[0,1]:-grouped[-1,1]]
    #grouped = grouped[1:-1]

    ### Freq of occurrence
    occurrence = grouped[:,0]
    # Count total number of occurrence
    unique, occurrence = np.unique(occurrence, return_counts=True)
    
    # Normalize respect to total time
    occurrence = occurrence / len(segmentation) * fs

    ### Coverage
    unique, coverage = np.unique(segmentation, return_counts=True)
    coverage = coverage / len(segmentation)
    
    # Check if microstate is not present
    if len(unique)!=n_states:
        absent = set(states) - set(unique)
        absent = list(absent)
        print(f'WARNING: states {absent} are absent...')
        for elem in absent:
            occurrence = np.insert(occurrence, elem, 0)
            coverage = np.insert(coverage, elem, 0)

    ### Mean duration
    duration = [np.mean(grouped[:,1][grouped[:,0]==state]) for state in range(n_states)]
    duration = np.array(duration) / fs * 1e3
    duration[isnull(duration)] = 0 #TODO: TO BE CHECKED
    
    ### GEV
    gfp = np.std(data, axis=0)
    map_corr = corr_vectors(data, maps[segmentation].T)
    # Compute gev for each timepoints
    gev = (gfp * map_corr) ** 2 / np.sum(gfp**2)
    # Compute gev for each state
    gev = [np.sum(gev[segmentation==state]) for state in states]
    gev = np.array(gev)
    
    return np.array([duration, coverage, occurrence, gev])

def get_t_matrix(data, maps):
    n_states = len(maps)
    states = np.arange(n_states)
    
    ### Get segmentation
    segmentation = back_fitting(data, maps)
    
    # Compute joint matrix
    q_matrix = np.zeros((n_states,n_states))

    for i in range(n_states):
        # init state
        s_in = segmentation[:-1] == states[i]
    
        for j in range(n_states):
            # ending state
            s_out = segmentation[1:] == states[j]
            # count occurrences
            q_matrix[i,j] = np.sum(s_in*s_out)
    
    q_matrix = q_matrix / q_matrix.sum(axis=1)[:,None]
    q_matrix[isnull(q_matrix)] = 0
    
    return q_matrix

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