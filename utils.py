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
            dat[file[:-4]] = pd.read_csv(os.path.join(folder,file))


if __name__ == "__main__":
    folder = 'C:\\Users\\User\\Documents\\GitHub\\NeuromatchGroup126A\\Data\\Forssmann_2017-11-04'
    load_session(folder)