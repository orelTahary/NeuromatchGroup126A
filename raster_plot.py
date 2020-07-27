def raster_plot(session_num,region,trial_num):
  '''
  Creates spike data and makes raster plot given session number, brain area and trial number 

  Args: 
        session_num: number of session (0,38)
        region: brain area that recordings have made
        trial_num: Trial number or given session >> can change should control by using ...['spks'].shape[1]

  Returns:
        neuron_spikes_times: A vector of when the neuron has fired given session
        raster plot

  
  '''

  session=alldat[session_num] # taking session
  spikes=session['spks'] # taking spikes
  brain_locs=session['brain_area'] # taking brain areas
  dt=session['bin_size'] # bin size

  uniq_brain_locs=np.unique(brain_locs) # locations that measurements made
  uniq_brain_locs_array=uniq_brain_locs==region
  uniq_brain_locs_array=uniq_brain_locs_array.astype(np.int)

  if np.sum(uniq_brain_locs_array)==0: # checking whether looked area has been used for measurement given session 
    print('This Region did not Used in This Session')
  else:
    neurons=np.where(brain_locs==region) # finding neurons that in the brain area
    neurons_spikes=spikes[neurons][:,trial_num] # taking spikes of given brain area
    neurons_spikes_times=neurons_spikes*(dt)*(np.arange(len(neurons_spikes[0]))) # determining spikes times
    

  for i in range(neurons_spikes_times.shape[0]):
    for j in range(neurons_spikes_times.shape[1]):
      if neurons_spikes_times[i,j]*100==2*j:
        neurons_spikes_times[i,j]=neurons_spikes_times[i,j]/2
  
  plt.eventplot(neurons_spikes_times,linelengths=0.3) # raster plot

    

  return neurons_spikes_times
