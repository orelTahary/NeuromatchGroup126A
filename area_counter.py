import numpy as np

regions,sessions_regions=regions_measured(0)

# brain groups defined in steinmetz dataset someone did the hardwork :D

brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                ] 
##############################################################

new_brain_groups=np.concatenate((brain_groups),axis=None)  # brain_groups concatenated
all_sessions_regions=np.concatenate((regions),axis=None) # all sessions concatenated
all_sess_reg_nums=[]

for i in range(len(new_brain_groups)):
  all_sess_reg_nums.append(np.count_nonzero(all_sessions_regions==new_brain_groups[i]))

all_sess_reg_nums=np.array(all_sess_reg_nums)
print(all_sess_reg_nums)
max_num=np.argmax(all_sess_reg_nums)
print(new_brain_groups[max_num])

brain_nums=[(all_sess_reg_nums[i],new_brain_groups[i]) for i in range(len(new_brain_groups))]
dtype = [('num', int), ('name', 'S10')]
a = np.array(brain_nums, dtype=dtype)   
np.sort(a,order='num')
