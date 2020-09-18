import numpy as np
from lorenz import get_data_lorenz_single
import matplotlib.pyplot as plt
from utils.misc import readPickle, savePickle, loadHDF5
import os
from model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate

from sklearn import preprocessing

# Settings for data generation
N = 1024
D = 3
t0 = 0.0
T = 40
delta_t = 0.025

# Parameters for learning
DIM_STOCHASTIC = 3  # dimensionality of latent space
EPOCHS = 1000

def generate_data():
    state0 = np.random.normal(0, 1, size=(N, D))
    state = np.zeros(shape=(N, int((T-t0)/delta_t), D))
    for n in range(N):
        state[n, :, :], _ = get_data_lorenz_single(state0[n, :], t0=t0, T=T, delta_t=delta_t)
    return state


def visualize_data(state, n_samples):
    print("state.shape=" + str(state.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for n in range(n_samples):
        ax.plot(state[n, :, 0], state[n, :, 1], state[n, :, 2], marker="x")
        plt.draw()
    plt.show()


def standardize_data(data):
    mean_X = data['train']['tensor'].mean(axis=(0, 1))
    std_X = data['train']['tensor'].std(axis=(0, 1))

    print(mean_X)
    data['train']['tensor'] = (data['train']['tensor']-mean_X)/std_X
    data['valid']['tensor'] = (data['valid']['tensor']-mean_X)/std_X
    data['test']['tensor'] = (data['test']['tensor']-mean_X)/std_X



def load_lorenz():
    curdir = os.path.dirname(os.path.realpath(__file__))
    fname = curdir+'/lorenz.pkl'
    if os.path.exists(fname):
        print 'Reloading dataset from ' + fname
        return readPickle(fname)[0]

    state = generate_data()

    shufidx = np.random.permutation(N)
    ntrain = int(0.7*N)
    ntest = int(0.2*N)
    nval = N - ntrain - ntest
    indices = {}
    indices['train'] = shufidx[:ntrain]
    indices['valid'] = shufidx[ntrain:ntrain+nval]
    indices['test'] = shufidx[ntrain+nval:]

    dataset = {}
    dataset['dim_observations'] = D
    for k in ['train', 'valid', 'test']:
        dataset[k] = {}
        dataset[k]['tensor'] = state[indices[k]]
        # dataset[k]['tensor_Z'] = []
        dataset[k]['mask'] = np.ones_like(dataset[k]['tensor'][:, :, 0])
    dataset['data_type'] = 'real'

    standardize_data(dataset)
    savePickle([dataset], curdir+'/lorenz.pkl')
    print 'Saving...'
    return dataset


def set_parameters(dataset):
    # sets up parameters
    params = readPickle('../default.pkl')[0]
    for k in params:
        print k, '\t',params[k]
    params['data_type'] = dataset['data_type']
    params['dim_observations'] = dataset['dim_observations']

    # The dataset is small, lets change some of the default parameters and the unique ID
    params['dim_stochastic'] = DIM_STOCHASTIC
    params['dim_hidden'] = 40
    params['rnn_size'] = 80
    params['epochs'] = EPOCHS
    params['batch_size'] = 200
    params['unique_id'] = params['unique_id'].replace('ds-100', 'ds-'+str(DIM_STOCHASTIC)).replace('dh-200','dh-40').replace('rs-600','rs-80')
    params['unique_id'] = params['unique_id'].replace('ep-2000','ep-1000').replace('bs-20','bs-200')

    # Create a temporary directory to save checkpoints
    params['savedir'] = params['savedir']+'/lorenz/'
    os.system('mkdir -p '+ params['savedir'])

    return params


def run_model():
    # Loads/generate data
    dataset = load_lorenz()
    params = set_parameters(dataset)
    print(params)
    print 'Dimensionality of the observations: ', dataset['dim_observations']
    print 'Data type of features:', dataset['data_type']
    for dtype in ['train','valid','test']:
        print 'dtype: ',dtype, ' type(dataset[dtype]): ', type(dataset[dtype])
        print [(k,type(dataset[dtype][k]), dataset[dtype][k].shape) for k in dataset[dtype]]
        print '--------\n'

    # Specify the file where `params` corresponding for this choice of model and data will be saved
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'

    print 'Checkpoint prefix: ', pfile
    dmm = DMM(params, paramFile=pfile)

    # savef specifies the prefix for the checkpoints - we'll use the same save directory as before
    savef = os.path.join(params['savedir'],params['unique_id'])
    savedata = DMM_learn.learn(dmm, dataset['train'], epoch_start=0,
                               epoch_end=params['epochs'],
                               batch_size=200,
                               savefreq=params['savefreq'],
                               savefile=savef,
                               dataset_eval=dataset['valid'],
                               shuffle=True)


def load_model_elbo():
    # Lets look at the statistics saved at epoch 40
    stats = loadHDF5('./chkpt/lorenz/DMM_lr-0_0008-dh-40-ds-'+str(DIM_STOCHASTIC)+'-nl-relu-bs-200-ep-1000-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP990-stats.h5')
    print [(k, stats[k].shape) for k in stats.keys()]
    plt.figure(figsize=(8, 10))
    plt.plot(stats['train_bound'][:, 0], stats['train_bound'][:, 1], '-o', color='g', label='Train')
    plt.plot(stats['valid_bound'][:, 0], stats['valid_bound'][:, 1], '-*', color='b', label='Validate')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Upper Bound on $-\log p(x)$')
    plt.show()


def sample_from_model():
    # Sampling from the model
    DIR = './chkpt/lorenz/'
    prefix = 'DMM_lr-0_0008-dh-40-ds-'+str(DIM_STOCHASTIC)+'-nl-relu-bs-200-ep-1000-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid'
    pfile = os.path.join(DIR, prefix + '-config.pkl')
    params = readPickle(pfile, quiet=True)[0]
    EP = '-EP990'
    reloadFile = os.path.join(DIR, prefix + EP + '-params.npz')
    print 'Model parameters in: ', reloadFile
    params['validate_only'] = True
    dmm_reloaded = DMM(params, paramFile=pfile, reloadFile=reloadFile)

    # (mu, logcov): parameters of emission distributions
    # z_vec = sample in latent space
    (mu, logcov), zvec = DMM_evaluate.sample(dmm_reloaded, T=40, nsamples=10)

    print("mu.shape=" + str(mu.shape))
    print("zvec.shape=" + str(mu.shape))

    visualize_data(mu, n_samples=10)
    plt.title("Mean trajectories")


    fig, axlist_x = plt.subplots(3, 1, figsize=(8, 10))
    nsamples = 10
    T = zvec.shape[1]
    SNUM = range(nsamples)
    for idx, ax in enumerate(axlist_x.ravel()):
         z = zvec[SNUM, :, idx]
         ax.plot(np.arange(T), np.transpose(z), '-*', label='Dim' + str(idx))
         ax.legend()
    ax.set_xlabel('Time')
    plt.suptitle('3 dimensional samples of latent space')
    plt.show()


def main():
    dataset = load_lorenz()
    visualize_data(dataset['train']['tensor'], n_samples=10)

    #run_model()
    #load_model_elbo()
    #sample_from_model()


if __name__ == "__main__":
    main()
