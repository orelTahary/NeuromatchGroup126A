import os
import numpy as np
import pandas as pd


def divide_to_trials(dat, bin_size):

    bin_ms = bin_size/1000
    clusters = dat['spikes.clusters']
    spks = dat['spikes.times']  
    intervals = dat['trials.intervals']
    spks_by_neuron = np.asarray([np.asarray(spks[clusters==cluster_num]) for cluster_num in dat['clusters.originalIDs']]) 
    max_interval = np.max(np.array([e-s for s,e in intervals]))
    spk_time_by_session = np.zeros((len(spks_by_neuron), len(intervals), int(max_interval//bin_ms)))
    for k, n in enumerate(spks_by_neuron):
        for j,inter in enumerate(intervals):
            spk_time_by_session[k,j,:] = np.histogram(n, bins=int(max_interval//bin_ms))[0]
    return spks_by_neuron, spk_time_by_session


def load_session(folder):
    dat = {}
    files = os.listdir(folder)
    for file in files:
        if file[-3:] == 'npy':
            dat[file[:-4]] = np.load(os.path.join(folder,file))
        if file[-3:] == 'tsv':
            dat[file[:-4]] = pd.read_csv(os.path.join(folder,file), sep='\t')
    return dat


def arrange_session(dat, bin_size=10):
    new_dat = {}
    # new_dat['mouse_name'] = 
    # new_dat['date_exp'] = 
    new_dat['bin_size'] = bin_size 
    clusters = dat['spikes.clusters']
    spks = dat['spikes.times']  # general, not devided by neurons
    new_dat['intervals'] = dat['trials.intervals']
    # an numpy array of arrays contaning spike times for each neuron
    new_dat['spike_times'], new_dat['spks'] = divide_to_trials(dat, bin_size)
    new_dat['brain_areas'] = np.asarray(dat['channels.brainLocation']['allen_ontology'].values)
    new_dat['contrast_right'] = dat['passiveVisual.contrastRight']
    new_dat['contrast_left'] = dat['passiveVisual.contrastLeft']
    new_dat['gocue'] = dat['trials.goCue_times']
    new_dat['response_times'] = dat['trials.response_times']
    new_dat['response'] = dat['trials.response_choice']
    new_dat['feedback_times'] = dat['trials.feedback_times']
    new_dat['feedback_type'] = dat['trials.feedbackType']
    new_dat['wheel'] = dat['wheel.position']
    new_dat['pupil'] = dat['eye.xyPos']
    # new_dat['lfp'] = dat[]
    # dat['brain_area_lfp'] = dat[]
    p2t = dat['spikes.amps']
    new_dat['trough_to_peak'] = np.asarray([np.asarray(p2t[clusters==cluster_num]) for cluster_num in dat['clusters.originalIDs']]) 
    new_dat['waveforms'] = dat['clusters.templateWaveforms']
    new_dat['lick_times'] = dat['licks.times']
    

    return new_dat


def random_session(data_fold):
    folds = os.listdir(data_fold)
    session_num = np.random.randint(len(folds))
    return os.path.join(data_fold, folds[session_num])


if __name__ == "__main__":
    data_fold = 'C:\\Users\\User\\Documents\\GitHub\\NeuromatchGroup126A\\Data'
    fold = random_session(data_fold)
    dat = load_session(fold)
    dat = arrange_session(dat, bin_size=10) # bin_size in milliseconds