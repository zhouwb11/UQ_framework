#!/usr/bin/env python

import os
import numpy as np
import matplotlib as mpl

def myrc():
    mpl.rc('legend', loc='best', fontsize=22)
    mpl.rc('lines', linewidth=3, color='r')
    mpl.rc('axes', linewidth=3, grid=True, labelsize=22)
    mpl.rc('xtick', labelsize=20)
    mpl.rc('ytick', labelsize=20)
    mpl.rc('font', size=20)
    mpl.rc('figure', figsize=(12, 9), max_open_warning=200)
    # mpl.rc('font', family='serif')

    return mpl.rcParams

def scale01ToDom(xx, dom):
    """Scaling set of inputs to a given domain, assuming \
       the inputs are in [0,1]^d
    """
    dim = xx.shape[1]
    xxsc = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(dim):
        xxsc[:, i] = xx[:, i] * (dom[i, 1] - dom[i, 0]) + dom[i, 0]

    return xxsc


def scaleDomTo01(xx, dom):
    """Scaling set of inputs from a given domain to [0,1]^d
    """
    dim = xx.shape[1]
    xxsc = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(dim):
        xxsc[:, i] = (xx[:, i] - dom[i, 0]) / (dom[i, 1] - dom[i, 0])

    return xxsc

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

def micf_join(mindex_list, cfs_list):
    nout = len(mindex_list)
    assert(nout == len(cfs_list))

    mindex_list_ = mindex_list.copy()
    cfs_list_ = cfs_list.copy()

    mindex0 = np.zeros((1, mindex_list_[0].shape[1]), dtype=int)
    cfs0 = np.zeros((1,))

    mindex_list_.append(mindex0)
    cfs_list_.append(cfs0)
    ### Get common set of multiindex and coefficients
    mindex_all = np.unique(np.concatenate(mindex_list_), axis=0)
    npc = mindex_all.shape[0]

    cfs_all = np.zeros((nout, npc))
    for j in range(nout):
        for k in range(npc):
            bb = np.sum(np.abs(mindex_list_[j]-mindex_all[k, :]), axis=1)
            ind = np.where(bb==0)[0]
            if len(ind) > 0:
                cfs_all[j, k] = cfs_list_[j][ind[0]]

    return mindex_all, cfs_all

# Again, this is not the most elegant multi-surrogate, since it has to write/read a lot of files as MCMC is being run
def multi_surrogate(q, results):
    uqtkbin = os.environ['UQTK_INS']+'/bin/'

    pccf_all = results['pcmi'][0]
    mindex_all = results['pcmi'][1]
    nout = len(mindex_all)
    if len(q.shape)==1:
        q = q.reshape(1,-1)

    nsam, ndim = q.shape
    np.savetxt('xdata.dat', q)

    y = np.empty((nsam, nout))
    for iout in range(nout):
        np.savetxt('mindex.dat', mindex_all[iout], fmt='%d')
        np.savetxt('pccf.dat', pccf_all[iout])

        command = uqtkbin +'pce_eval -x PC_mi -f pccf.dat -s LU -r mindex.dat > pce_eval.log'
        os.system(command)
        y[:, iout] = np.loadtxt('ydata.dat')

    if nsam == 1:
        y=y.reshape(-1,)

    return y

def multisens(mindex_all, cfs_all, sens_type='total'):
    uqtkbin = os.environ['UQTK_INS']+'/bin/'

    nx = cfs_all.shape[0]
    ndim = mindex_all.shape[1]
    allsens = np.zeros((nx, ndim))

    np.savetxt('mi', mindex_all, fmt='%d')
    for ix in range(nx):
        np.savetxt('cfs', cfs_all[ix, :])
        os.system(uqtkbin+'pce_sens -m mi -f cfs > pce_sens.log')
        if sens_type == 'total':
            allsens[ix, :] = np.loadtxt('totsens.dat')
        elif sens_type == 'main':
            allsens[ix, :] = np.loadtxt('mainsens.dat')

    return allsens
