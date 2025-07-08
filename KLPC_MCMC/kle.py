#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

from utils import myrc

myrc()

class KLE(object):
    def __init__(self):

        self.built = False

        self.mean = None
        self.eigval = None
        self.kl_modes = None
        self.weights2 = None
        self.rel_var = None
        self.xi = None

        return

    def build(self, data, plot=False):
        ngrid, nens = data.shape
        self.mean = np.mean(data, axis=1)
        cov = np.cov(data)

        # Set trapesoidal rule weights
        self.weights2 = np.ones(ngrid)
        self.weights2[0] = 0.5
        self.weights2[-1] = 0.5
        weights = np.sqrt(self.weights2)

        cov_sc = np.outer(weights, weights) * cov

        self.eigval, eigvec = np.linalg.eigh(cov_sc)
        self.eigval = self.eigval[::-1]
        eigvec = eigvec[:, ::-1]

        self.kl_modes = eigvec / weights.reshape(-1, 1) # ngrid, neig
        # Negative fix for small eigenvalues
        self.eigval[self.eigval<1.e-14] = 1.e-14



        tmp = self.kl_modes * np.sqrt(self.eigval)
        self.rel_var = (np.cumsum(tmp * tmp, axis=1) + 0.0) / (np.diag(cov).reshape(-1, 1) + 0.0) #ngrid, neig

        self.built = True

        self.xi = self.project(data)


        if plot:
            self.plot_eig()
            self.plot_klmodes()

        return

    def project(self, data):
        assert(self.built)
        xi = np.dot(data.T - self.mean, self.kl_modes * self.weights2.reshape(-1, 1)) / np.sqrt(self.eigval) #nens, neig

        return xi

    def eval(self, xi=None, neig=None):
        assert(self.built)

        ngrid = self.kl_modes.shape[0]

        if xi is None:
            xi = self.xi

        if neig is None:
            neig = ngrid

        data_kl = self.mean + np.dot(xi[:, :neig] * np.sqrt(self.eigval[np.newaxis, :neig]), self.kl_modes[:, :neig].T)
        data_kl = data_kl.T # now ngrid, nens

        # data_kl = self.mean + np.dot(np.dot(xi[:, :neig], np.diag(np.sqrt(self.eigval[:neig]))), self.kl_modes[:, :neig].T)
        # data_kl = data_kl.T # now data_kl is ngrid, nens just like data

        return data_kl

    def plot_klmodes(self, imodes=None):
        assert(self.built)

        ngrid = self.kl_modes.shape[0]
        if imodes is None:
            imodes = range(ngrid)

        _ = plt.figure(figsize=(12,9))
        plt.plot(range(ngrid), self.mean, label='Mean')
        for imode in imodes:
            plt.plot(range(ngrid), self.kl_modes[:, imode], label='Mode '+str(imode+1))
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('KL Modes')
        plt.legend()
        plt.savefig('KLmodes.png')
        plt.close()


    def plot_eig(self):
        assert(self.built)

        _ = plt.figure(figsize=(12,9))
        plt.plot(range(1, self.eigval.shape[0]+1),self.eigval, 'o-')
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('Eigenvalue')
        plt.savefig('eig.png')
        plt.gca().set_yscale('log')
        plt.savefig('eig_log.png')
        plt.close()

