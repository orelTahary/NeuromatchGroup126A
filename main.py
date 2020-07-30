import numpy as np
import utils as u
alldat = u.load_data()

accuracies = np.load('accuracies.npy',allow_pickle='TRUE').item()
all_sessions = range(len(alldat))
for i_session in range(3): #all_sessions:
    bareas = np.unique(alldat[i_session]['brain_area'][:-1])
    session_move_acc = []
    session_dir_acc = []
    for area in bareas:
        area_move_acc = [accuracies[key] for key in accuracies if (str(i_session) in key) and ('M' in key) and (area in key)]
        session_move_acc.append(area_move_acc)
        area_dir_acc = [accuracies[key] for key in accuracies if (str(i_session) in key) and ('D' in key) and (area in key)]
        session_dir_acc.append(area_dir_acc)
        print(f'acurracy in session {i_session} in {area} was {area_move_acc}')

orel = 1