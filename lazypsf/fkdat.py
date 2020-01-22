import subprocess
import random
from math import factorial as fct
from math import atan2
import math
import itertools
import os
from multiprocessing import Pool
from multiprocessing import cpu_count

from astropy.nddata.utils import Cutout2D as cut
from astropy.io import fits
import numpy as np
from scipy import ndimage
import sep



def BG(mod, X, Y):
    """
    Using a predefined model, create a source kernel
    If model = none, will use basic Gaussian
    """
    #sort this out at some point
    kernel=np.zeros((30,30))
    if mod == None:
        for i in range(30):
            for j in range(30):
                kernel[i][j] = 1/np.sqrt(2*np.pi*6.25) * np.exp(-(((i-13)**2/12.5)+((j-13)**2/12.5)))
    kernel = kernel/np.sum(kernel)
    return(kernel)


def do_inject(im, X, Y, p_mod, flux):
    """
    Ry Cutter 2020

    ~~~~~
    for now I'm lazy
    ~~~~~
    """

    kernel = BG(p_mod, X, Y)
    kx = len(kernel)
    ky = len(kernel[0])
    if p_mod is None:
        for i in range(kx):
            for j in range(ky):
                try:
                    im[X-int(kx/2)+i][Y-int(ky/2)+j] += kernel[i][j] * flux
                except:
                    continue


    #print(np.sum(kernel))
    #print(kernel)
    return(im)


def make_noise(image):
    image = np.random.normal(3,0.35, size=image.shape)
    return(image)

def fake_data(X_size, Y_size, pix_size = None, num_sources = 500, pos = None, fluxes = None, noise = None, PSF = None, outname='out.fits', overwrite = False):
    """
    Ry Cutter 2020

    ~~~~~
    Make your own Data
    X_size = width of the field in pixels
    Y_size = height of the field in pixels
    pix_size = 
    num_sources = number of sources to inject
    pos = position of sources [(x1,y1), (x2,y2)... (xn,yn)] if none random 
    fluxes = flux of sources [F1, F2, Fn] if none random
    noise = TBD
    PSF = Point Spread Function model. To be inserted by another function.
    ~~~~~
    """
    
    #Start by checking logics
    if pos is not None and len(pos)!=num_sources:
        raise ValueError('Number of source positions should be equal to number of sources')
    if pos is not None:
        for i in pos:
            if 0 > i[0] > X_size:
                raise ValueError('Source position outside of X axis range') 
            if 0 > i[1] > Y_size:
                raise ValueError('Source position outside of Y axis range')
    else:
        Xses = np.random.randint(1,X_size, size=num_sources)
        Yses = np.random.randint(1,Y_size, size=num_sources)
        pos = []
        for i in range(num_sources):
           pos.append([Xses[i], Yses[i]])

    if fluxes is not None and len(pos)!=num_sources:
        raise ValueError('Number of source positions should be equal to number of sources')
    elif fluxes is None:
         fluxes = np.random.randint(5000,68000, size=num_sources)
         #fluxes = np.random.normal(5000,68000, size=num_sources)

    image = np.zeros((X_size, Y_size))
    
    if noise != None:
        image = make_noise(image)

    #Inject them sources boooiii
    for i in range(len(pos)):
        image = do_inject(image,pos[i][0],pos[i][1],PSF,fluxes[i]) 
    

    #Make fits
    fits.writeto(outname, image, overwrite=overwrite)
    return(outname)





     
