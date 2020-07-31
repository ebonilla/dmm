# loads results from syntetic example
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth']=2.5
mpl.rcParams['lines.markersize']=8
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['text.latex.preamble']= ['\usepackage{amsfonts}','\usepackage{amsmath}']
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize']=20
mpl.rcParams['legend.fontsize']=20

import numpy as np
from utils.misc import getConfigFile, readPickle, displayTime, loadHDF5
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate

import os

#Lets look at the statistics saved at epoch 40
stats = loadHDF5('./chkpt/synthetic/DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP30-stats.h5')
print [(k,stats[k].shape) for k in stats.keys()]
plt.figure(figsize=(8,10))
plt.plot(stats['train_bound'][:,0],stats['train_bound'][:,1],'-o',color='g',label='Train')
plt.plot(stats['valid_bound'][:,0],stats['valid_bound'][:,1],'-*',color='b',label='Validate')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Upper Bound on $-\log p(x)$')
plt.show()

# Sampling from the model
DIR    = './chkpt/synthetic/'
prefix = 'DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid'
pfile  =  os.path.join(DIR,prefix+'-config.pkl')
params = readPickle(pfile, quiet=True)[0]
EP     = '-EP30'
reloadFile  =  os.path.join(DIR,prefix+EP+'-params.npz')
print 'Model parameters in: ',reloadFile
params['validate_only'] = True
dmm_reloaded  = DMM(params, paramFile = pfile, reloadFile = reloadFile)

(mu, logcov), zvec = DMM_evaluate.sample(dmm_reloaded, T= 10)

fig,axlist_x = plt.subplots(3,1,figsize=(8,10))
SNUM         = 0
for idx, ax in enumerate(axlist_x.ravel()):
    mu_x = mu[SNUM,:,idx]
    ax.plot(np.arange(mu_x.shape[0]), mu_x, '-*', label = 'Dim'+str(idx))
    ax.legend()
ax.set_xlabel('Time')
plt.suptitle('3 dimensional samples')
plt.show()
