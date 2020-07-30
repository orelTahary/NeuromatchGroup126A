import numpy as np
import utils as u
import plot_utils as pu
from matplotlib import rcParams 
from matplotlib import pyplot as plt
import statsmodels.api as sm #for GLM
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_validate, KFold, cross_val_predict, cross_val_score 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, SCORERS, make_scorer
from sklearn.decomposition import PCA 

# TODO:
# - load data
# - assiciate neuron with brain regions
# - dreate design matrix
# - divide into train set and validation
# - prepare all GLM inputs
# - apply GLM on brain region
# - test results and get accuracy
# - compare accuracy of different brain regions
# - plot results

alldat = u.load_data()
num_sessions = len(alldat)
accuracies = {}
for i_session in range(num_sessions):
	# dat['brain_region'] = u.brain_areas(dat)
	areas = np.unique(alldat[i_session]['brain_area'])[:-1]
	print(f'in session {i_session} training model on areas {areas}')
	for area in areas:
		accuracies[f'session {i_session} {area} D'] = u.chosen_direction(alldat, i_session, 1 , areas,'P','D')
		accuracies[f'session {i_session} {area} M'] = u.chosen_direction(alldat, i_session, 1 , areas,'P','M')

orel = 1