import os
import numpy as np
import pandas as pd

def load_session(folder):
    dat = {}
    files = os.listdir(folder)
    for file in files:
        if file[-3:] == 'npy':
            dat[file[:-4]] = np.load(os.path.join(folder,file))
        if file[-3:] == 'tsv':
            dat[file[:-4]] = pd.read_csv(os.path.join(folder,file), sep='\t')
    return dat


def arrange_session(dat):
    new_dat = {}
    # new_dat['mouse_name'] = 
    # new_dat['date_exp'] = 
    new_dat['spks'] = dat['spikes.times']
    new_dat['brain_areas'] = np.asarray(dat['channels.brainLocation']['allen_ontology'].values)
    new_dat['contrast_right'] = dat['passiveVisual.contrastRight']
    new_dat['contrast_Left'] = dat['passiveVisual.contrastLeft']
    new_dat['gocue'] = dat['trials.goCue_times']
    new_dat['response_times'] = dat['trials.response_times']
    new_dat['response'] = dat['trials.response_choice']
    new_dat['feedback_times'] = dat['trials.feedback_times']
    new_dat['feedback_type'] = dat['trials.feedbackType']
    new_dat['wheel'] = dat['wheel.position']
    new_dat['pupil'] = dat['eye.xyPos']
    # new_dat['lfp'] = dat[]
    # dat['brain_area_lfp'] = dat[]
    new_dat['trough_to_peak'] = dat['spikes.amps']
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
    dat = arrange_session(dat)