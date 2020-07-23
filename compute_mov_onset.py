import numpy as np
from matplotlib import rcParams 
from matplotlib import pyplot as plt

# Upload data -> THOSE LINES HAVE TO BE CHANGED
wheel=dat['wheel']
gocue=dat['gocue']
feed_time=dat['feedback_time']
sample_stim=50 # sample where the stimulus occured (-> after 500ms, bin time=10 ms)

# Settable parameters 
# # Threshold for considering the 'wheel as turned' 
threshold=5
# the trials that you want to consider
trials_interval=np.array(range(0,340))  

# Initialization
n_trials=len(trials_interval)
sample_stim=50

## --- Settable parameter:  
# Threshold of when considered the 'wheel turned' 
threshold=5 
# Trials that you want to check
trials_interval=np.array(range(0,340))  

# Initialization
n_trials=len(trials_interval)
mov_onset=[] #time in ms of when the mice moves for the first time the wheel given the threshold above -> NO MOVEMENT value = 50000
trial_NoWheel=[] #indeces of the trials where the mice did not turn the wheel 
direction=[] #<0 choose right (turn left?); =0 No turn; >0 choose left (turn right?);
onset_gocue=[] #<0 before gocue; >0 after gocue

for i in trials_interval:
  sample_go=int(stimulus+gocue[i]*100)
  sample_feed=np.minimum(int(sample_stim+feed_time[i]*100), 250) # compute the time when the reward is provided
  wheel_trial=wheel[0,i,sample_stim:sample_feed] # only considering the interval from the stimulus to the feedback to look for the movement onset
  idx=np.argwhere(abs(wheel_trial)>threshold) # index when the wheel is more turned than the threshold

  if len(idx)==0:
    mov_onset.append(50000) 
    direction.append(0)
  else:
    mov_onset.append(idx[0]*10) #in ms
    onset_gocue.append(np.sign(mov_onset[j]-sample_go))
    direction.append(np.sign(wheel_trial[idx[0]]))
   
plt.hist(mov_onset)

# print(len(resp[resp==0]))
# print(len(mov_onset))
# print(trial_NoWheel)
 
# plt.plot(mov_onset)
# print(mov_onset)
