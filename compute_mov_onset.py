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
trials_interval=np.array([5, 6, 7, 8, 9])#np.array(range(0,340))  

# Initialization
n_trials=len(trials_interval)
mov_onset=np.zeros(n_trials)  #sample where the wheel is above the threshold for the first time; No turn-> mov_onset = 0
direction=np.zeros(n_trials) #<0 chosoe right (move left?); =0 No turn; >0 choose left (turn right?)
onset_gocue=np.zeros(n_trials) #<0 before gocue; =0 No turn, >0 after gocue

j=0

for i in trials_interval:
  sample_go=int(stimulus+gocue[i]*100)
  sample_feed=np.minimum(int(sample_stim+feed_time[i]*100), 250) # compute the time when the reward is provided
  wheel_trial=wheel[0,i,sample_stim:sample_feed] # only considering the interval from the stimulus to the feedback to look for the movement onset
  idx=np.argwhere(abs(wheel_trial)>threshold) # index when the wheel is more turned than the threshold

  if len(idx)==0:
    mov_onset[j]=0 
    direction[j]=0
    onset_gocue[j]=0
  else:
    mov_onset[j]=idx[0]+sample_stim
    direction[j]=np.sign(wheel_trial[idx[0]]) 
    onset_gocue[j]=np.sign(mov_onset[j]-sample_go)
  j=j+1
  # print(mov_onset)
  # ax=plt.subplot(2,5,i-9)
  # plt.plot(wheel[0,i,:],color='b')
  # plt.axvline(mov_onset[i],color='k')
 
# plt.plot(mov_onset)
# print(mov_onset)
