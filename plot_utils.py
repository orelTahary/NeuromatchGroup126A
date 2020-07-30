from matplotlib import rcParams 
from matplotlib import pyplot as plt


def plot_direction(direction, mov_onset):
    plt.plot(direction)
    new_direction=direction[mov_onset !=0]
    new_direction[new_direction == -1] = 0 #to have 0 1 instead of -1 1
    plt.plot(new_direction)
    plt.show()

def plot_trial(dat, trial_num):
    
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

