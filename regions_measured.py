import numpy as np

def regions_measured(session_num):
  '''
  Checks which brain areas were recorded from given session.

  Arg:
  session_num: number of the session (0,38)

  Output: 
  regions: gives every recorded areas session by session
  sessions_regions: brains areas that recording had been taken

  '''
  regions=[]
  for i in range(len(alldat)):
    regions.append(np.unique(alldat[i]['brain_area']))

  sessions_regions=regions[session_num]

  print(sessions_regions)

  return regions,sessions_regions
