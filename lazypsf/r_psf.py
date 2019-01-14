import subprocess
import random
import math
import itertools
import os

from astropy.nddata.utils import Cutout2D as cut
from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt

#use the correct call for sextractor
try:
    sex = 'sextractor'
    subprocess.call(sex, stderr=subprocess.DEVNULL)
except:
    sex = 'sex'
    subprocess.call(sex, stderr=subprocess.DEVNULL) 

root_config = os.path.dirname(os.path.realpath(__file__))+'/config' #path to config files


def FWHM_shape(cat, im_dat, chop_x= 3, chop_y= 3):
    """
    Ryan Cutter 2019

    Returns FWHM at different points on the CCD
    """
    X_s, Y_s = im_dat.shape
    F = np.zeros((chop_x, chop_y)) #FWHM at specific part of CCD
    E = np.zeros((chop_x, chop_y)) #Elongation
    T = np.zeros((chop_x, chop_y)) #Angle of source
    X_b = np.linspace(0, X_s, chop_x+1)
    Y_b = np.linspace(0, Y_s, chop_y+1)
    for i in range(chop_x):
        for j in range(chop_y):
            FWHMs = []
            ELONs = []
            Thetas = []
            for Lin in open(cat, 'r'):
                tmp = Lin.split()
                X = float(tmp[0])
                Y = float(tmp[1])
                if X_b[i] < X < X_b[i+1] and Y_b[j] < Y < Y_b[j+1]:
                    FWHMs.append(float(tmp[2]))
                    ELONs.append(float(tmp[3]))
                    Thetas.append(float(tmp[6]))
            F[i][j] = np.mean(FWHMs)
            E[i][j] = np.mean(ELONs)
            T[i][j] = np.mean(Thetas)
    return(F, E, T)



def source_model(bb, flux, FWHM, data, ELON=0, Theta=90):
    """
    Ryan Cutter 2019


    Inject a fake source into data. Assuming the data has the same
    shape as the bound box (bb). The fake source should meet the 
    requirements described by Flux, FWHM, ELon and Theta.


    returns 
    ------------
    Small data object with a fake modelled source added in.
    """
    if len(bb) != 4:
        raise ValueError('Bounding box needs 4 boundaries!')
    if data.shape[0] != (bb[0]+bb[2]) and data.shape[1] != (bb[1]+bb[3]):
        print(data.shape, [bb[0]+bb[2],bb[1]+bb[3]])
        raise ValueError('Bounding box does not match cutout shape')
    Theta = Theta * 0.0174533
    #  Using the fact that FWHM/2 = 2.355*sig

    FWHMY = FWHM #FWHM semi_minor
    sig_y = FWHM / (2 * 2.355)

    if ELON != 0:
        FWHMX = FWHM * ELON #FWHM semi_major 
        sig_x = FWHMX / (2*2.2355)
    else:
        FWHMX = FWHM
        sig_x = sig_y



    pi = math.pi
    f_half_ratio = 1/np.sqrt(2*pi*sig_y**2)* math.exp(-(FWHM/2)**2/(2*sig_y**2))
    f_max = flux*f_half_ratio*2

    if FWHMX > bb[2]: #Stop source stretching out of box
        FWHM_TX = bb[2]
    else:
        FWHM_TX = FWHMX

    if FWHMY > bb[3]: #Stop source stretching out of box
        FWHM_TY = bb[3]
    else:
        FWHM_TY = FWHMY

    if FWHMX > bb[0]:
        FWHM_BX = bb[0]
    else:
        FWHM_BX = FWHMX

    if FWHMY > bb[1]:
        FWHM_BY = bb[1]
    else:
        FWHM_BY = FWHMY

    FLUXES = [] #This will be the flux of each injected pixel 
    for i in range(-int(round(FWHM_BX)), int(round(FWHM_TX))):
        for j in range(-int(round(FWHM_BY)), int(round(FWHM_TY))):
            FLUXES.append(
                          1 / np.sqrt(2 *pi * sig_x * sig_y ) * math.exp( 
                            -( ( i**2 / (2*sig_x**2)) + ( j**2 / (2 * sig_y**2 ))  
                )
               )
              )

    FLUXES = [(float(i)/max(FLUXES))*f_max for i in FLUXES]
    k = -1
    for i in range(-int(round(FWHM_BX)), int(round(FWHM_TX))):
        for j in range(-int(round(FWHM_BY)), int(round(FWHM_TY))):
            k+=1
            x = i
            y = j #x,y coords of non rotated source

            if Theta != 0:
                theta_xy=np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)

                x_n = r*np.cos(theta_xy+Theta)
                y_n = r*np.sin(theta_xy+Theta)

            else:
                x_n = x
                y_n = y

            data[bb[0]+int(round(x_n)), bb[1]+int(round(y_n))] = FLUXES[k] + data[bb[0]+int(round(x_n)), bb[1]+int(round(y_n))]

    return(data)


def inject_fake_source(flux, image_dat, X, Y, FWHM, ELON=0, Theta = 0, cut_size=50):
    """
    Ryan Cutter 2019

    Using the source characteristics (FWHM, Theta, and Elon)
    Inject a fake source into image with flux at position
    X, Y

    reuturns
    -----------
    Data object with injected source
    """
    X_M, Y_M = image_dat.shape #Maximum points in data
    if X > X_M or Y > Y_M:
        raise ValueError('Injection out of bounds')

    CUT = cut(image_dat, (Y, X), (cut_size, cut_size))

    L_bx = round(cut_size/2)
    L_by = round(cut_size/2) #expected cut out lengths above centre point
    if cut_size%2 == 0:
        L_tx = round(cut_size/2)
        L_ty = round(cut_size/2)
    else: 
        L_tx = round(cut_size/2) +1
        L_ty = round(cut_size/2) +1 #expected cut out lengths below centre point

    if X + L_tx > X_M: #If expected cutout exceeds upper boarder correct length 
        L_tx = X_M - X 
    if Y + L_ty > Y_M: 
        L_ty = Y_M - Y 

    if X - L_bx < 0: #If expected cutout exceeds lower boarder correct length
        L_bx =  X
    if Y - L_by < 0: 
        L_by =  Y

    out_data = source_model([L_bx, L_by, L_tx, L_ty],flux, FWHM, CUT.data, ELON, Theta)
    fits.writeto('test.fits', out_data, overwrite = True)

    image_dat[X - L_bx: X + L_tx, Y - L_by: Y + L_ty] = out_data
    fits.writeto('test2.fits', image_dat, overwrite=True)
    return(image_dat)


def inject_fake_stars(image, chop_x, chop_y, n_stars, flux, write_cat = False):
    """
    Ryan Cutter 2019

    Runs sextractor on the inserted image
    to get PSF stats. Uses the stats to inject
    flux_dist is the flux distribution

    returns
    -----------
    data object with n_stars injected with matced PSF
    """



    image_dat = fits.getdata(image)
    X_s, Y_s = image_dat.shape #data shape
    talk = [sex ,image,'-c',root_config+'/r_psf.sex' , '-CATALOG_NAME' , 'r.cat',
            '-PARAMETERS_NAME', root_config+'/r_psf.param', '-STARNNW_NAME', 
            '-FILTER_NAME', root_config+'/default.conv'
            ]
    print('Making PSF cat')
    subprocess.call(talk, stderr=subprocess.DEVNULL)
    F, E, T = FWHM_shape('r.cat', image_dat, chop_x, chop_y) #get PSF stats

    X_b = np.linspace(0, X_s, chop_x+1) #define stat bounds
    Y_b = np.linspace(0, Y_s, chop_y+1)

    if write_cat == True:
        CAT = open('injection.cat', 'w')

    for stars in range(n_stars):
        #print(stars)
        X_in = np.random.randint(0, X_s)
        Y_in = np.random.randint(0, Y_s)
        for i in range(chop_x):
            for j in range(chop_y):
                if X_b[i] <= X_in <= X_b[i+1] and Y_b[j] <= Y_in <= Y_b[j+1]:
                    kern_i = i
                    kern_j = j #j position in kern

        F_k = F[kern_i][kern_j]
        E_k = E[kern_i][kern_j]
        T_k = T[kern_i][kern_j]

        if write_cat == True:
            CAT.write(str(X_in)+' '+str(Y_in)+' '+str(flux)+'\n')


        print(F)
        print(E)
        image_dat = inject_fake_source(flux, image_dat, X_in, Y_in, F_k, E_k, T_k, cut_size=80) #rounding is strange for cutout2D, stick to even cut_size
        #subprocess.call(['ds9', 'test.fits'])
    subprocess.call(['rm', 'r.cat'])
    return(image_dat)


def PSF_kern(image, chunk=50):
    """
    Ryan Cutter 2019

    Makes a Guassian kernel for a given image
    """
    pi = math.pi

    image_dat = fits.getdata(image)
    X_s, Y_s = image_dat.shape
    X_m = round(X_s/2)
    Y_m = round(Y_s/2) # img_dat mid points
    kern = np.zeros((X_s, Y_s))
    
    talk = [sex ,image,'-c',root_config+'/r_psf.sex' , '-CATALOG_NAME' , 'r.cat',
            '-PARAMETERS_NAME', root_config+'/r_psf.param', '-STARNNW_NAME', 
            '-FILTER_NAME', root_config+'/default.conv'
            ]
    print('Making PSF cat')
    subprocess.call(talk, stderr=subprocess.DEVNULL)
    
    FWHM, ELON, Theta = FWHM_shape('r.cat', image_dat, 1, 1)
    Theta = Theta[0][0] * 0.0174533

    FWHMY = FWHM[0][0]
    sig_y = FWHM / (2 * 2.355)
    ELON=0
    if ELON != 0:
        FWHMX = FWHM[0][0] * ELON[0][0]
        sig_x = FWHMX / (2*2.2355)
    else:
        FWHMX = FWHMY
        sig_x = sig_y

    WEIGHTS = []
    for i in range(-chunk, chunk):
        for j in range(-chunk, chunk):
            WEIGHTS.append(1 / np.sqrt(2 *pi * sig_x * sig_y )
                          * math.exp( -(( i**2 / (2*sig_x**2)) + ( j**2 / (2 * sig_y**2 ))
                )
               )
              )

    WEIGHTS = [(float(i)/max(WEIGHTS)) for i in WEIGHTS]
    k = -1
    for i in range(-chunk, chunk):
        for j in range(-chunk, chunk):
            k +=1
            x = i
            y = j #x,y coords of non rotated source

            if Theta != 0:
                theta_xy=np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)

                x_n = r*np.cos(theta_xy+Theta)
                y_n = r*np.sin(theta_xy+Theta)

            else:
                x_n = x
                y_n = y

            #noise = int(np.random.normal(M_dat, std_dat))
            kern[X_m+int(round(x_n))][Y_m+int(round(y_n))] = WEIGHTS[k]

    subprocess.call(['rm', 'r.cat'])
    return(kern)


def inject_basic(image, chop_x, chop_y, n_stars, out_name='inject.fits', flux_dist = 9000, write_cat = True):
    if flux_dist==0:
        print('TO DO, make flux distribution')
    elif flux_dist != 0:
        flux =  flux_dist
    dat = inject_fake_stars(image, chop_x, chop_y, n_stars, flux, write_cat)
    fits.writeto(out_name, dat, overwrite=True)
