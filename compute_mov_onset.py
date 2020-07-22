import numpy as np
from matplotlib import rcParams 
from matplotlib import pyplot as plt

# Upload data -> THOSE LINES HAVE TO BE CHANGED
wheel=dat['wheel']
gocue=dat['gocue']
feed_time=dat['feedback_time']
sample_stim=50 # sample where the stimulus occured (-> after 500ms, bin time=10 ms)

# Settable parameters: Threshold for considering the 'wheel as turned' and the trials that you want to check
threshold=5
trials_interval=np.array([5, 6, 7, 8, 9])#np.array(range(0,340))  

# Initialization
n_trials=len(trials_interval)
mov_onset=np.zeros(n_trials)
j=0

for i in trials_interval:
  sample_feed=np.minimum(int(sample_stim+feed_time[i]*100), 250) # compute the time when the reward is provided
  wheel_trial=wheel[0,i,sample_stim:sample_feed] # only considering the interval from the stimulus to the feedback to look for the movement onset
  idx=np.argwhere(abs(wheel_trial)>threshold) # index when the wheel is more turned than the threshold

  if len(idx)==0:
    mov_onset[j]=0 # if there is no wheel data > threshold -> mov_onset = 0
  else:
    mov_onset[j]=idx[0]+sample_stim
  j=j+1
  # print(mov_onset)
  # ax=plt.subplot(2,5,i-9)
  # plt.plot(wheel[0,i,:],color='b')
  # plt.axvline(mov_onset[i],color='k')
 
# plt.plot(mov_onset)
# print(mov_onset)
