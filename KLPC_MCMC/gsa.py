#!/usr/bin/env python


import numpy as np
from utils import *


###########################################
#  ____     ___    ____     ___    _      #
# / ___|   / _ \  | __ )   / _ \  | |     #
# \___ \  | | | | |  _ \  | | | | | |     #
#  ___) | | |_| | | |_) | | |_| | | |___  #
# |____/   \___/  |____/   \___/  |_____| #
#                                         #
###########################################


class smethod(object):
    def __init__(self,dom,sens_names):
        self.dom=dom
        self.dim=dom.shape[0]
        self.sens = dict((k, [None] * self.dim) for k in self.sens_names)
        self.sens_ready = dict((k, [False] * self.dim) for k in self.sens_names)


class sobol(smethod):
    ## Main and total are based on Saltelli 2010; it computes joint in the total sense!
    ##
    ## Initialization
    def __init__(self,dom):
        print("Initializing SOBOL")
        self.sens_names=['main', 'total', 'jointt']
        smethod.__init__(self,dom,self.sens_names)

    def sample(self,ninit,samplepar=None):
        print("Sampling SOBOL")

        sam1=scale01ToDom(np.random.rand(ninit,self.dim),self.dom)
        sam2=scale01ToDom(np.random.rand(ninit,self.dim),self.dom)

        xsam=np.vstack((sam1,sam2))

        for id in range(self.dim):
            samid=sam1.copy()
            samid[:,id]=sam2[:,id]
            xsam=np.vstack((xsam,samid))

        self.nsam=xsam.shape[0]
        self.sens_ready['main']   = True
        self.sens_ready['total']  = True
        self.sens_ready['jointt'] = True

        return xsam

    def compute(self,ysam,computepar=None):
        ninit=self.nsam//(self.dim+2)
        y1=ysam[ninit:2*ninit]
        var=np.var(ysam[:2*ninit])
        si=np.zeros((self.dim,))
        ti=np.zeros((self.dim,))
        jtij=np.zeros((self.dim,self.dim))

        for id in range(self.dim):
            y2=ysam[2*ninit+id*ninit:2*ninit+(id+1)*ninit]-ysam[:ninit]
            si[id]=np.mean(y1*y2)/var
            ti[id]=0.5*np.mean(y2*y2)/var
            for jd in range(id):
                y3=ysam[2*ninit+id*ninit:2*ninit+(id+1)*ninit]-ysam[2*ninit+jd*ninit:2*ninit+(jd+1)*ninit]
                jtij[id,jd]=ti[id]+ti[jd]-0.5*np.mean(y3*y3)/var

        self.sens['main']=si
        self.sens['total']=ti
        self.sens['jointt']=jtij.T

        return self.sens


