import numpy as np
from lorenz import get_data_lorenz_coupled
import matplotlib.pyplot as plt
from utils.misc import readPickle, savePickle, loadHDF5
import os

from parse_args import params

from model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate


def standardize_data(data):
    mean_X = data['train']['tensor'].mean(axis=(0, 1))
    std_X = data['train']['tensor'].std(axis=(0, 1))

    print(mean_X)
    data['train']['tensor'] = (data['train']['tensor']-mean_X)/std_X
    data['valid']['tensor'] = (data['valid']['tensor']-mean_X)/std_X
    data['test']['tensor'] = (data['test']['tensor']-mean_X)/std_X


def generate_data(n, t0, T, delta_t):
    """
    :param n: Number of observations (sequences)
    :param d: dimensionality
    :param t0:
    :param T:
    :param delta_t:
    :return state: n x T x 9
    """
    dim = 9  # Coupled example is 9d
    state0 = np.random.normal(0, 1, size=(n, dim))
    state = np.zeros(shape=(n, int((T-t0)/delta_t), dim))
    for n in range(n):
        state[n, :, :], _ = get_data_lorenz_coupled(state0[n, :], t0=t0, T=T, delta_t=delta_t)
        print("sample " + str(n) + " done")
    return state


def load_lorenz_coupled():
    n = 1000
    delta_t = 0.025
    t0 = 0.0
    T = 100.0

    curdir = os.path.dirname(os.path.realpath(__file__))
    fname = curdir+'/lorenz_coupled.pkl'

    if os.path.exists(fname):
        print 'Reloading dataset from ' + fname
        return readPickle(fname)[0]

    state = generate_data(n=n, t0=t0, T=T, delta_t=delta_t)

    shufidx = np.random.permutation(n)
    ntrain = int(0.7*n)
    ntest = int(0.2*n)
    nval = n - ntrain - ntest
    indices = {}
    indices['train'] = shufidx[:ntrain]
    indices['valid'] = shufidx[ntrain:ntrain+nval]
    indices['test'] = shufidx[ntrain+nval:]

    dataset = {}
    dataset['dim_observations'] = state.shape[2]
    for k in ['train', 'valid', 'test']:
        dataset[k] = {}
        dataset[k]['tensor'] = state[indices[k]]
        # dataset[k]['tensor_Z'] = []
        dataset[k]['mask'] = np.ones_like(dataset[k]['tensor'][:, :, 0])
    dataset['data_type'] = 'real'

    standardize_data(dataset)
    savePickle([dataset], fname)
    print 'Saving...'
    return dataset


def visualize_data(state, n_samples):
    print("state.shape=" + str(state.shape))

    # 3d systems
    fige = plt.figure()
    figt = plt.figure()
    figo = plt.figure()

    axe = fige.gca(projection='3d')
    axt = figt.gca(projection='3d')
    axo = figo.gca(projection='3d')
    for n in range(n_samples):
        axe.plot(state[n, :, 0], state[n, :, 1], state[n, :, 2])
        axt.plot(state[n, :, 3], state[n, :, 4], state[n, :, 5])
        axo.plot(state[n, :, 6], state[n, :, 7], state[n, :, 8])

    plt.draw()
    plt.show()

    # x trajectories
    # T = 40
    # t = np.arange(start=0, stop=T, step=delta_t)
    # L_t = t.size
    #
    # plt.plot(t, x_e[:L_t])
    # plt.show()
    #
    # plt.plot(t, x_t[:L_t])
    # plt.show()
    #
    # plt.plot(t, X[:L_t])
    # plt.show()


def main():
    dataset = load_lorenz_coupled()
    visualize_data(dataset['train']['tensor'], n_samples=10)


if __name__ == "__main__":
    # main()
    print(params)

