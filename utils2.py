import os, requests
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import and setting for the logistic regression
import statsmodels.api as sm #for GLM
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_validate, KFold, cross_val_predict, cross_val_score 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, SCORERS, make_scorer
from sklearn.decomposition import PCA 


def load_data():
    #@title Data loading
    fname = []
    for j in range(3):
        fname.append('steinmetz_part%d.npz'%j)
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")

    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

    return alldat

def brain_areas(dat, nareas = 7):
    # groupings of brain regions
    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                    ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]

    NN = len(dat['brain_area']) # number of neurons
    barea = nareas * np.ones(NN, ) # last one is "other"
    
    for j in range(nareas):
        barea[np.isin(dat['brain_area'], brain_groups[j])] = j # assign a number to each region
    
    neuron_region = np.array(['other' for i in barea])
    for area in barea:
        neuron_region[barea==area] = regions[int(area)]
    
    return neuron_region


# FUNCTION TO COMPUTE MOVEMNT ONSET, DIRECTION 
def extract_behaviours(alldat, session_number,threshold=5):
  sample_stim=50
  j=0
  dat = alldat[session_number]
  wheel=dat['wheel']
  gocue=dat['gocue']
  feed_time=dat['feedback_time']

  # Trials that you want to check
  trials_interval=np.array(range(0, dat['spks'].shape[1]))  
  print(len(trials_interval))
  # Initialization
  n_trials=len(trials_interval)
  mov_onset=np.zeros(n_trials)
  direction=np.zeros(n_trials) #<0 choose right (turn left?); =0 No turn; >0 choose left (turn right?);
  onset_gocue=[] #<0 before gocue; >0 after gocue

  for i in trials_interval:
    sample_go=int(sample_stim+gocue[i]*100)
    sample_feed=np.minimum(int(sample_stim+feed_time[i]*100), dat['spks'].shape[2]) # compute the time when the reward is provided
    wheel_trial=wheel[0,i,sample_stim:sample_feed] # only considering the interval from the stimulus to the feedback to look for the movement onset
    idx=np.argwhere(abs(wheel_trial)>threshold) # index when the wheel is more turned than the threshold
    
    if len(idx)==0:
      mov_onset[j]=0
      direction[j]=0
    else:
      mov_onset[j]=idx[0] #*10 in msec
      # onset_gocue.append(np.sign(mov_onset[-1]-sample_go))
      direction[j]=np.sign(wheel_trial[idx[0]])
    j=j+1
  return direction, mov_onset

  # print(len(resp[resp==0]))
  # print(len(mov_onset))
  # print(trial_NoWheel)
  direction,mov_onset=extract_behaviours(alldat, 11)
  plt.plot(direction)
  new_direction=direction[mov_onset !=0]
  new_direction[new_direction == -1] = 0 #to have 0 1 instead of -1 1
  plt.plot(new_direction)
  # print(mov_onset)
  # ax=plt.subplot(2,5,i-9)
  # plt.plot(wheel[0,i,:],color='b')
  # plt.axvline(mov_onset[i],color='k')
 
# plt.plot(mov_onset)
# print(mov_onset)


# LOGISTIC REGRESSION TO PREDICT DIRECTION OF TURNING FROM SPIKE TRAINS OF CHOOSEN BRAIN REGION
def get_behavior(alldat, session_number, nareas, brain_groups, sample_stim=50):
    window_size= 100 #number of samples considered after the stimulus onset 
  
  #take neural data of the session
  dat=alldat[session_number] # taking session
  nneurons = len(dat['brain_area']) 
  spikes = dat['spks']

  #take data only from the brain regions choosen
  barea = nareas * np.ones(nneurons, ) # last one is "other"
  for j in range(nareas):
    barea[np.isin(dat['brain_area'], brain_groups[j])] = j # assign a number to each region

  #compute the behavioural data  
  direction,mov_onset=extract_behaviours(alldat, session_number, 5)

  #take data in a time-window after the stimulus time
  window_spks = np.zeros((spikes.shape[0],spikes.shape[1], window_size))
  for i in range(0,len(mov_onset)):
    if  mov_onset[i] in range(1,window_size-1):
      window = np.arange(sample_stim,sample_stim+window_size) 
      window_spks [:,i,:] = spikes[:, i, window]
    else:  
      window = np.arange(0,window_size-1) #if there no movement we  keep the first values (..they will be deleted later..)  

  #Remove the trial where there is no movement in the window time 
  idx_notzero = np.argwhere(mov_onset!=0)
  idx_less= np.argwhere(mov_onset<(window_size-1)) #mov_onset = 0, these trials are deleted in behavioral and neural data
  idx_mov=np.intersect1d(idx_notzero, idx_less)
  
  # direction array only when there is movement
  new_direction=direction[idx_mov]
  new_direction[new_direction == -1] = 0 #to have 0 1 instead of -1 1 (0=turn left, 1=turn right)
  
  # movement onset time only when there is movement and cathegorized in interval 
  n_level=5
  new_mov_onset=np.round(mov_onset[idx_mov]//(window_size/n_level))+1

  return new_direction, new_mov_onset
def design_matrix():
def apply_pca():

def chosen_direction(alldat, session_number, nareas, brain_groups, flag, flag_y, sample_stim=50):
  #session_number --> to retrieve data from alldat
  #nareas --> int, how many areas we want (4 means the first four of the list..)
  #brain_groups --> which brain area that you want to use
  #flag -> different ways to solve overfitting problem (R or P or others for doing nothing)

  
  # plt.scatter(np.array([range(0,len(new_mov_onset))]),new_mov_onset)

  # Select which array do you want to predict
  if flag_y=='D':
    y=new_direction
  else :
    y=new_mov_onset
  
  for j in range(nareas):
    new_dat = window_spks[:,idx_mov,:] #delete where there no movement
    new_dat_onearea = new_dat[barea == j] #one area

    if flag=='R': #Randomly picking electrods from the whole area
      idx_random=np.random.choice(new_dat_onearea.shape[0],size=18, replace=False) #int(new_dat_onearea.shape[0]/3)#random selection of some electrods
      #print(len(idx_random))
      new_dat_onearea=new_dat_onearea[idx_random,:,:]

    #Creation design matrix
    for k in range(new_dat_onearea.shape[0]):
      if k==0:
        matrix_el=new_dat_onearea[k,:,:]
      else:
        matrix_el=np.hstack((matrix_el, new_dat_onearea[k,:,:]))
        
    print('Before matrix:',matrix_el.shape)
    
    if flag=='P': #PCA
      new_shape_dat_onearea = np.reshape(new_dat_onearea, (new_dat_onearea.shape[0],-1)) #row=neuron, column=trial*time
      mean_sub=np.mean(new_shape_dat_onearea, axis=1)[:, np.newaxis]
      print(mean_sub.shape)
      new_shape_dat_onearea = new_shape_dat_onearea - mean_sub
      total_var_explained=0
      n_PC=2

      while total_var_explained<=0.8:
        model = PCA(n_components = n_PC).fit(new_shape_dat_onearea.T) 
        W = model.components_
        total_var_explained=model.explained_variance_ratio_.sum()
        print('N component:',n_PC,' Variance explained:',total_var_explained)
        n_PC=n_PC+4

      n_PC=n_PC-4
      print('Model:',W.shape,'input PCA:',new_shape_dat_onearea.shape)
      no_matrix_el=W@new_shape_dat_onearea
      matrix_el=np.zeros([new_dat_onearea.shape[1],n_PC*100])
      for pc in range(n_PC):
        for tr in range(new_dat_onearea.shape[1]):
          matrix_el[tr,pc*window_size:(window_size-1)+(pc*window_size)]=no_matrix_el[pc,tr*window_size:(window_size-1)+tr*window_size]
      # matrix_el=np.reshape(matrix_el, (new_dat_onearea.shape[1],-1))
      print(matrix_el.shape)

    # Apply logistic regression and cross validation
    if new_dat_onearea.shape[0] != 0:
      kf = 4 #k fold
      logistic = cross_validate(LogisticRegression(penalty = "l2", max_iter = 300, class_weight='balanced'), matrix_el, y=y, cv=kf,  return_train_score=True, return_estimator=True) # k=8 crossvalidation

      # print(logistic['estimator'][1].coef_) #take the coeff of each model
      print('training mean accuracy =',logistic['train_score'].mean())
      print('test mean accuracy =',logistic['test_score'].mean())
    
    return logistic['test_score'].mean()

# Brain areas considered
# areas=[["MOs"]]#["GPe","PL","SNr","ACA","VPL","ILA","ZI","MOs"]]

# Find sessions with the brain areas considered
# session=print_sessions_with_area(areas[0])

#Logistic regression function
# chosen_direction(21,1,areas,'P','D')


# IF OVERFITTING: (DONE) 
#               SUBSAMPLE NEURONS FROM DIFFERENT REGION (RANDOMLY) (DONE)
#               REGULARIZATION (DONE)
#               PCA (DONE?)
# CROSS VALIDATION WITHIN THE SAME SESSION (DONE)
# PLAY WITH PARAMETERS
#                     : LogisticRegression(regolarizator, solver), PC_components, number of random electrodes (=PC), brain areas(session), window_size 

# NOTE: This code works also with mov_onset, but it can predict if the mov_onset_time falls in one time-interval 
# -> I divided the window in N intervals(<10) and it predicts in which interval will happen


if __name__ == "__main__":
  pass
