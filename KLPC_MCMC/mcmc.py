#!/usr/bin/env python

import sys
import numpy as np



class AMCMC(object):
    def __init__(self):
        super(AMCMC, self).__init__()
        return

    def setParams(self, param_ini=None, cov_ini=None,
                  t0=100, tadapt=1000,
                  gamma=0.1, nmcmc=10000):
        self.param_ini = param_ini
        self.cov_ini = cov_ini
        self.t0 = t0
        self.tadapt = tadapt
        self.gamma = gamma
        self.nmcmc = nmcmc

    # def setData_calib(self, xd, yd):
    #     self.xd = xd # list of conditions
    #     self.yd = yd # list of arrays per condition

    # Adaptive Markov chain Monte Carlo
    def run(self, logpostFcn, postInfo):
        cdim = self.param_ini.shape[0]            # chain dimensionality
        cov = np.zeros((cdim, cdim))   # covariance matrix
        samples = np.zeros((self.nmcmc, cdim))  # MCMC samples
        logpm = np.zeros((self.nmcmc, 1))  # log posteriors
        na = 0                        # counter for accepted steps
        sigcv = self.gamma * 2.4**2 / cdim
        samples[0] = self.param_ini                  # first step
        # print(samples[0])  # 111
        p1 = -logpostFcn(samples[0], postInfo)  # NEGATIVE logposterior
        pmode = p1  # record MCMC 'mode', which is the current MAP value (maximum posterior)
        cmode = samples[0]  # MAP sample
        acc_rate = 0.0  # Initial acceptance rate

        logpm[0] = -p1

        # Loop over MCMC steps
        for k in range(self.nmcmc - 1):

            # Compute covariance matrix
            if k == 0:
                Xm = samples[0]
            else:
                Xm = (k * Xm + samples[k]) / (k + 1.0)
                rt = (k - 1.0) / k
                st = (k + 1.0) / k**2
                cov = rt * cov + st * np.dot(np.reshape(samples[k] - Xm, (cdim, 1)), np.reshape(samples[k] - Xm, (1, cdim)))
            if k == 0:
                propcov = self.cov_ini
            else:
                if (k > self.t0) and (k % self.tadapt == 0):
                    propcov = sigcv * (cov + 10**(-8) * np.identity(cdim))

            # Generate proposal candidate
            u = np.random.multivariate_normal(samples[k], propcov)
            p2 = -logpostFcn(u, postInfo)
            # print(u, p1, p2)  # 111
            pr = np.exp(p1 - p2)
            # Accept...
            if np.random.random_sample() <= pr:
                samples[k + 1] = u
                na = na + 1  # Acceptance counter
                p1 = p2
                logpm[k+1] = -p1
                if p1 <= pmode:
                    pmode = p1
                    cmode = samples[k + 1]
            # ... or reject
            else:
                samples[k + 1] = samples[k]
                logpm[k+1] = logpm[k]

            acc_rate = float(na) / self.nmcmc

            if((k + 2) % (self.nmcmc / 10) == 0) or k == self.nmcmc - 2:
                print('%d / %d completed, acceptance rate %lg' % (k + 2, self.nmcmc, acc_rate))
                np.savetxt('chain.txt', samples)

        mcmc_results = {'chain' : samples, 'mapparams' : cmode, 'maxpost' : pmode, 'accrate' : acc_rate, 'logpostm' : logpm}

        return mcmc_results




# Function that computes log-posterior
# given model parameters
def logpost(modelpars, lpinfo):

    # !!!!! This is a quick hack: if parameters are outside prior [-1,1], then we give very low value so that sample is rejected
    for par in modelpars:
        # print(par)
        if abs(par)>1:
            return -1.e+80

    # Model prediction
    ypred = lpinfo['model'](modelpars, lpinfo['otherpars'])
    # print('ypred:')  # 111
    # print(ypred)
    # Data
    ydata = lpinfo['yd']
    nd = len(ydata)
    # print('ydata:')
    # print(ydata)

    if lpinfo['ltype'] == 'classical':
        lpostm = 0.0
        for i in range(nd):
            for yy in ydata[i]:
                lpostm -= 0.5 * (ypred[i]-yy)**2/lpinfo['lparams']['sigma'][i]**2
                # print('ypred:')
                # print(ypred[i])
                # print('111:')
                # print(lpostm)
                lpostm -= 0.5 * np.log(2 * np.pi)
                # print('222:')
                # print(lpostm)
                lpostm -= np.log(lpinfo['lparams']['sigma'][i])
                # print('333:')
                # print(lpostm)
    else:
        print('Likelihood type is not recognized. Exiting')
        sys.exit()

    return lpostm
