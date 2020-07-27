import os, requests
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_data():
    #@title Data loading
    fname = []
    for j in range(3):
        fname.append('steinmetz_part%d.npz'%j)
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")

    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

    return alldat

def brain_areas(dat, nareas = 7):
    # groupings of brain regions
    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                    ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]

    NN = len(dat['brain_area']) # number of neurons
    barea = nareas * np.ones(NN, ) # last one is "other"
    
    for j in range(nareas):
        barea[np.isin(dat['brain_area'], brain_groups[j])] = j # assign a number to each region
    
    dat['brain_region'] = np.array(['other' for i in barea])
    for area in barea:
        dat['brain_region'][barea==area] = regions[int(area)]
    
    return dat

def get_mov_onset(dat, stim=50, threshold=5):
    wheel=dat['wheel']
    gocue=dat['gocue']
    feed_time=dat['feedback_time']
    # trials_interval=np.array(range(0,feed_time.shape[0])) 
    
    n_trials = dat['spks'].shape[1]
    mov_onset=[] #time in ms of when the mice moves for the first time the wheel given the threshold above -> NO MOVEMENT value = 50000
    trial_NoWheel=[] #indeces of the trials where the mice did not turn the wheel 
    direction=[] #<0 choose right (turn left?); =0 No turn; >0 choose left (turn right?);
    onset_gocue=[] #<0 before gocue; >0 after gocue

    for trial in range(n_trials):
        sample_go=int(stim+gocue[trial]*100)
        sample_feed=np.minimum(int(stim+feed_time[trial]*100), 250) # compute the time when the reward is provided
        wheel_trial=wheel[0,trial,stim:sample_feed] # only considering the interval from the stimulus to the feedback to look for the movement onset
        idx=np.argwhere(abs(wheel_trial)>threshold)

        if len(idx)==0:
            mov_onset.append(50000) 
            trial_NoWheel.append(trial)
            direction.append(0)
        else:

            mov_onset.append(idx[0]+stim)
            onset_gocue.append(np.sign(mov_onset[-1]-sample_go))
            direction.append(np.sign(wheel_trial[idx[0]]))
    
    dat['mov_onset'] = np.array(mov_onset) # in bins 
    dat['trials_NoWheel'] = np.array(trial_NoWheel) # trial indices
    dat['mov_direction'] = np.array(direction)
    dat['onset_gocue'] = np.array(onset_gocue)
    
    return dat

def plot_mov_onset(dat, trial_num):
    
    wheel = dat['wheel'][0,trial_num]
    if dat['mov_direction'][trial_num] < 0:
        direction = 'right'
    elif dat['mov_direction'][trial_num] > 0:
        direction = 'left'
    else:
        direction = 'no move'
    
    plt.plot(dat['wheel'][0,trial_num, :])
    plt.axvline(50, color='purple', label="stimulus onset")
    plt.axvline(dat['mov_onset'][trial_num], color='orange', label="movement onset")
    plt.axvline(dat['gocue'][trial_num]*100, color='green', label="go cue")
    plt.axvline(dat['feedback_time'][trial_num]*100,color='yellow', label="feedback time")
    plt.legend()
    plt.title(f'trial num {trial_num}, direction: {direction}')

    plt.show()

def spikes_before_mov_onset(dat, n_bins=50):
    
    spks_trials_moving = np.delete(dat['spks'],dat['trials_NoWheel'], axis=1)
    spks_b4_mov = np.zeros(spks_trials_moving.shape, dtype=bool)
    mov_onset = np.delete(dat['mov_onset'],dat['trials_NoWheel'])
    n_trials = spks_b4_mov.shape[1]
    for trial in range(n_trials):
        onset = mov_onset[trial]
        spks_b4_mov[:,trial,int(onset-n_bins):int(onset)] = True
    
    dat['spks_b4_mov'] = spks_trials_moving[spks_b4_mov]
    return dat

if __name__ == "__main__":

    alldat = load_data()
    session_num = 1
    dat = alldat[1]
    dat = brain_areas(dat)
    dat = get_mov_onset(dat)
    dat = spikes_before_mov_onset(dat)
    
    orel = 1
