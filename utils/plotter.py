import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from utils.utils import back_fitting, smoothing, get_names

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'lines.linewidth': 2})

def plot_maps(maps, info, figsize=None, letters=False, save=False):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_channels, n_maps)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    """
    if figsize is None:
        figsize = (2 * len(maps), 2)
        
    if letters:
        names = get_names(len(maps))
        
    plt.figure(figsize=figsize)
    for i, map in enumerate(maps):
        ax = plt.subplot(1, len(maps), i + 1)
        plot_topomap(map, info, axes=ax, show=False, contours=0)
        
        if letters:
            plt.title(names[i])
        else:
            plt.title('%d' % i)
            
    if save:
        plt.savefig('figures/group_maps.svg', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def plot_gfp(data, sigma, min_peak_dist,
             maps = None, cmap = 'plasma',
             show_segmentation=False, show_maps=True, figsize=(18,6),
             fact_w = 10, fact_h = 3.5, tup=1000, trange = 300):
    
    ### Prepare data
    # Smoothing
    data_smooth = smoothing(data, sigma=sigma)
    # GFP
    gfp = np.std(data, axis=0)
    gfp_smooth = np.std(data_smooth, axis=0)
    # GFP peaks
    peaks, _ = find_peaks(gfp_smooth, distance=min_peak_dist)
    n_peaks = len(peaks)
    tmp_max = gfp[tup:tdw].max()

    ### Create figure
    fig, ax = plt.subplots(figsize=figsize)

    tdw = tup+trange
    times = np.arange(trange) / fs
    tmp_peaks = peaks[(peaks>tup)*(peaks<tdw)]

    plt.plot(gfp[tup:tdw], label='gfp')
    plt.plot(gfp_smooth[tup:tdw], label='smooth')

    plt.plot((tmp_peaks-tup), gfp_smooth[tmp_peaks], 'o', ms=10, c='r')

    plt.xlim([0,tdw-tup])
    plt.ylim([0,tmp_max*1.2])

    plt.xlabel('t')
    plt.ylabel('GFP')

    ### Add maps inset
    if show_maps:
        bbox = ax.get_position()
        scal = bbox.width / trange
        for i, peak in enumerate(tmp_peaks):
            left, bottom, width, height = [bbox.x0+scal*(peak-tup)-bbox.width/fact_w/2, bbox.y0+bbox.height*0.75, bbox.width/fact_w, bbox.height/fact_h]
            # Create axes and plot maps
            ax2 = fig.add_axes([left, bottom, width, height])
            mne.viz.plot_topomap(data_smooth[:,peak], pos=dataset.info, axes=ax2, show=False)
            # Add line from GFP peak to correspondent map
            ax.plot([peak-tup, peak-tup], [gfp_smooth[peak], tmp_max*0.8], color='k', linestyle='-', linewidth=1)
    
    ### Add segmentation
    if (maps is not None) and show_segmentation==True:
        # Compute segmentation
        segmentation = back_fitting(data_smooth, maps)
        n_states = len(maps)
        # Color area under GFP
        cmap = plt.cm.get_cmap(cmap, n_states)
        for state, color in zip(range(n_states), cmap.colors):
            ax.fill_between(times, gfp_smooth[tup:tdw], color=color, where=(segmentation[tup:tdw] == state), interpolate=True)
        
    ### Show image 
    plt.show()

def plot_statistics(stats_all, cmap = 'plasma'):
    labels = ['Duration [ms]', 'Coverage', 'Occurrence [1/s]', 'GEV']
    tasks = ['resting', 'high', 'medium', 'low']
    
    n_task = len(tasks)
    n_measure = len(labels)
    n_states = stats_all.shape[-1]
    cmap = plt.cm.get_cmap(cmap, n_task)
    
    fig = plt.figure(figsize=(15,10))

    for measure in range(n_measure):
        ax = plt.subplot(2,2,measure+1)

        for task in range(n_task):
            plt.boxplot(stats_all[:,task,measure], positions=np.arange(n_states)*n_states+task,
                    notch=True, patch_artist=True,
                    boxprops=dict(color=cmap(task), facecolor='w') )

        plt.xlabel('state')
        plt.ylabel(labels[measure])
        plt.xticks(np.arange(n_states)*n_states + 1., get_names(n_states))

    #plt.legend(loc=2, ncol=2)
    plt.tight_layout()
    plt.show()
    
    
'''  
def plot_segmentation(segmentation, data, times, polarity=None):
    """Plot a microstate segmentation.

    Parameters
    ----------
    segmentation : list of int
        For each sample in time, the index of the state to which the sample has
        been assigned.
    times : list of float
        The time-stamp for each sample.
    polarity : list of int | None
        For each sample in time, the polarity (+1 or -1) of the activation.
    """
    gfp = np.std(data, axis=0)
    if polarity is not None:
        gfp *= polarity

    n_states = len(np.unique(segmentation))
    plt.figure(figsize=(6 * np.ptp(times), 2))
    cmap = plt.cm.get_cmap('plasma', n_states)
    plt.plot(times, gfp, color='black', linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        plt.fill_between(times, gfp, color=color,
                         where=(segmentation == state))
    norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.xlabel('Time (s)')
    plt.title('Segmentation into %d microstates' % n_states)
    plt.autoscale(tight=True)
    plt.tight_layout()
'''