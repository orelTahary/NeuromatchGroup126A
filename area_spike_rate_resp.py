import numpy as np
from matplotlib import rcParams 
from matplotlib import pyplot as plt

def area_spike_rater_resp(session_num,area):

  '''
  area_spike_rater_resp(session_num,area)

  desc: shows spike rate according to left and right responses given trial number and brain area

  args:

  session_num: session number defined by Steinmetz et. al.
  
  area: Spike recording area

  output: None

  '''

  dat=alldat[session_num]
  neurons_to_see=np.where(dat['brain_area']==area)
  response = dat['response'] # I will come up with a def can take this as an arg

  dt = dat['bin_size'] # binning at 10 ms
  NT = dat['spks'].shape[-1]

  stacked_neurons=np.stack([dat['spks'][i,:,:] for i in range(len(to_see[0]))])


  plt.plot(dt * np.arange(NT), 1/dt * stacked_neurons[:,response>=0].mean(axis=(0,1))); # left responses # I will try to add boolean logic in this code
  plt.plot(dt * np.arange(NT), 1/dt * stacked_neurons[:,response<0].mean(axis=(0,1))); # right responses
  

  plt.xlabel('time (sec)');
  plt.ylabel('firing rate (Hz)');
  plt.legend(['left response','right response'])

# area_spike_rater_resp(11,'ACA')

