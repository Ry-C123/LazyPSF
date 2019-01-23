import subprocess
import random
import math
import itertools
import os

from astropy.nddata.utils import Cutout2D as cut
from astropy.io import fits
import numpy as np
from scipy import ndimage
#import matplotlib.pyplot as plt
try:
    import pyfftw.interfaces.numpy_fft as fft
except:
    print('For a faster performance install pyfftw')
    print(' ')
    print(' ')
    print(' ')
    import numpy.fft as fft  #Use if you don't have pyfftw


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
            '-PARAMETERS_NAME', root_config+'/r_psf.param',
            '-FILTER_NAME', root_config+'/r_psf.conv'
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


def clean_norm_psf (psf_ar, clean_fact = 0.25):
    """
    Normalises the psf required for ZOGY, will remove any values smaller than the clean factor 
    to avoid clutter by precision
    """
    ysize, xsize = psf_ar.shape
    assert ysize == xsize

    hsize = ysize/2

    if xsize % 2 == 0:
        x = np.arange(-hsize,hsize)
    else:
        x = np.arange(-hsize,hsize)

    xx, yy = np.meshgrid(x,x, sparse=True)
    psf_ar[(xx**2+yy**2)>hsize**2] = 0

    if clean_fact != 0:
     mask_clean = (psf_ar < (np.amax(psf_ar)*clean_fact))
     psf_ar[mask_clean]=0

    psf_ar_norm = psf_ar / np.sum(psf_ar)

    return(psf_ar_norm)

def get_psf(image):
    """ 
    Finds the psf using PSFex.
    """

    sexcat = image.replace('.fits', '_PSFCAT.fits')
    sexcat = sexcat.replace('./Zoutput/','') 
    talk = [sex ,image,'-c',root_config+'/default.sex' , '-CATALOG_NAME' , sexcat,
           '-PARAMETERS_NAME', root_config+'/default.param', '-STARNNW_NAME', 
           root_config+'/default.nnw', '-FILTER_NAME', root_config+'/default.conv']

    print('Making PSF catalog')
    subprocess.call(talk, stderr=subprocess.DEVNULL)

    outcat = sexcat.replace('_PSFCAT.fits', '.psfexcat')
 
    print('Modelling PSF for '+image)
    talk2 = ['psfex', sexcat, '-c', root_config+'/psfex.conf', '-OUTCAT_NAME', outcat]
    subprocess.call(talk2,stderr=subprocess.DEVNULL)
    with fits.open(sexcat.replace('.fits','.psf')) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data

    dat = data[0][0][:]
    return(dat, header, sexcat, outcat)


def psf_map(dat, header, const, xl, yl, xc, yc, slices):
    """
    Ryan Cutter 2018 (copied from zogyp)

    Maps the PSF data to a kernel for convolution
    """
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_samp = header['PSF_SAMP']


    psf_size_config = header['PSFAXIS1']
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    if psf_size % 2 == 0:
        psf_size += 1
    psf_samp_update = float(psf_size) / float(psf_size_config)

    ysize_fft = yl
    xsize_fft = xl

    xcenter_fft, ycenter_fft = xsize_fft/2, ysize_fft/2


    psf_centre = np.zeros((ysize_fft,xsize_fft), dtype='float32')
      #The PSF found by PSFex in the centre of the cutout

    x = (xc - polzero1) / polscal1
    y = (yc - polzero2) / polscal2

    if slices == 1:
        psf = dat[0]
    else:
        if poldeg == 2:
            psf = dat[0] + dat[1] * x + dat[2] * x**2 + dat[3] * y + dat[4] * x * y + dat[5] * y**2
        elif poldeg == 3:
            psf = dat[0] + dat[1] * x + dat[2] * x**2 + dat[3] * x**3 + \
                  dat[4] * y + dat[5] * x * y + dat[6] * x**2 * y + \
                  dat[7] * y**2 + dat[8] * x * y**2 + dat[9] * y**3

    psf_resized = ndimage.zoom(psf, psf_samp_update)
    psf_resized_norm = clean_norm_psf(psf_resized, const)
    psf_hsize = math.floor(psf_size/2)

    ind = [slice(int(ycenter_fft-psf_hsize), int(ycenter_fft+psf_hsize+1)),
           slice(int(xcenter_fft-psf_hsize), int(xcenter_fft+psf_hsize+1))]

    psf_centre[ind] = psf_resized_norm
    return(psf_centre)

def PSF_kern2(image, clean_psf = 0.25):
    """
    Ryan Cutter 2019

    Compiles the two principal functions to return a PSF kernel

    returns
    --------------
    n by m array kernel to convolve with input image (of size n by m)
    """
    dat, header, sexcat, outcat = get_psf(image)
    yl, xl = fits.getdata(image).shape
    xc = xl/2
    yc = yl/2 
    kern = psf_map(dat, header, clean_psf, xl, yl, xc, yc, 0)
    return(kern)


def psfex_source_model(flux, dat, header, const, xc, yc, DATA):
    """
    Ryan Cutter 2019

    Finds the PSF using PSFex and applies it to inject a fake source
    """
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_samp = header['PSF_SAMP']

    
    psf_size_config = header['PSFAXIS1']
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    if psf_size % 2 == 0:
        psf_size += 1
    psf_samp_update = float(psf_size) / float(psf_size_config)

    x = (xc - polzero1) / polscal1
    y = (yc - polzero2) / polscal2

    ysize_fft = DATA.shape[0]
    xsize_fft = DATA.shape[1]

    xcenter_fft, ycenter_fft = xsize_fft/2, ysize_fft/2


    psf_centre = np.zeros((ysize_fft,xsize_fft), dtype='float32')


    if poldeg == 2:
        psf = dat[0] + dat[1] * x + dat[2] * x**2 + dat[3] * y + dat[4] * x * y + dat[5] * y**2
    elif poldeg == 3:
        psf = dat[0] + dat[1] * x + dat[2] * x**2 + dat[3] * x**3 + \
              dat[4] * y + dat[5] * x * y + dat[6] * x**2 * y + \
              dat[7] * y**2 + dat[8] * x * y**2 + dat[9] * y**3


    psf_resized = ndimage.zoom(psf, psf_samp_update)
    psf_resized_norm = clean_norm_psf(psf_resized, const)
    psf_hsize = math.floor(psf_size/2)

    ind = [slice(int(ycenter_fft-psf_hsize), int(ycenter_fft+psf_hsize+1)),
           slice(int(xcenter_fft-psf_hsize), int(xcenter_fft+psf_hsize+1))]

    psf_centre[ind] = psf_resized_norm
    out_dat = np.zeros((psf_centre.shape[0], psf_centre.shape[1]))    

    if psf_centre.shape != DATA.shape:
        raise ValueError('PSF dimensions do not match injection dimensions')

    
    norm = np.sum(psf_centre)

    for i in range(len(psf_centre)):
        for j in range(len(psf_centre[1])):
            psf_centre[i][j]= psf_centre[i][j]/norm * flux
            out_dat[i][j] = psf_centre[i][j]+DATA[i][j]


    return(out_dat)

def inject_fake_source(flux, image_dat, PSF_mod, X, Y, FWHM = 0, ELON=0, Theta = 0, cut_size=50, PSF_dat = 0):
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

    if PSF_mod == 1:
        out_data = psfex_source_model(flux, PSF_dat[0], PSF_dat[1], 0.005 , X, Y, CUT.data)
    else:
        out_data = source_model([L_bx, L_by, L_tx, L_ty],flux, FWHM, CUT.data, ELON, Theta)

    fits.writeto('test.fits', out_data, overwrite = True)

    image_dat[X - L_bx: X + L_tx, Y - L_by: Y + L_ty] = out_data
    #fits.writeto('test2.fits', image_dat, overwrite=True)
    return(image_dat)


def get_flux_dist(image):
    """
    Ryan Cutter 2019
    
    This function takes the input data 
    and creates a flux distribution
    representative of the image
    
    Better for finding realistic recovery stats
    
    returns
    ----------------
    flux distribution params
    """
    
    cat = 'FLUX.cat'
    fluxes=[] #Collection of all the flux in the image 
    talk = [sex ,image,'-c',root_config+'/r_psf.sex' , '-CATALOG_NAME' , cat,
            '-PARAMETERS_NAME', root_config+'/r_psf.param',
            '-FILTER_NAME', root_config+'/r_psf.conv'
            ]
    print('Making flux distribution catalog')
    subprocess.call(talk, stderr=subprocess.DEVNULL)
    for Lin in open(cat, 'r'):
        flux = float(Lin.split()[5])
        if flux > 0:
            fluxes.append(flux)
    Len = len(fluxes)
    MAX = np.mean(fluxes)+np.std(fluxes)*2
    MIN = np.mean(fluxes)-np.std(fluxes)/2
    if MIN < 0:
        MIN=0
    if Len < 500:
    	raise ValueError('Print, not enough sources to build reliable flux distribution')
    	
    #Create distribution bins
    bins = 5 * (math.ceil(Len/500)) #minimum of 5 bins
    flux_step = (MAX - MIN)/bins
    #print("flux_step  " + str(flux_step))
    dist = np.zeros((int(bins),2))
    TOTAL = 0 #for normalizing 
    for i in range(int(bins)):
        LL = MIN + (flux_step * i) #Lower limit
        UP = MIN + (flux_step * (i+1))  #Upper limit
        dist[i][1] = LL
        for j in fluxes:
            if i == 0:
                if j <= UP:
                    dist[i][0] += 1
                    TOTAL +=1
            elif i == int(bins)-1:
                if j > LL:
                    dist[i][0] += 1
                    TOTAL +=1
            else:
                if LL < j <= UP:
                    dist[i][0] += 1
                    TOTAL += 1

    for i in range(int(bins)):
        dist[i][0] = dist[i][0]/TOTAL

    subprocess.call(['rm', cat])

    return(dist)
    
def zp_dist(mag_l, mag_u, zp):
    """
    Ryan Cutter 2019
    
    This function creates a uniform distribution 
    based on the mag limits and given zero_point
    Where mag_l is fainter:     mag_l > mag_u

    Good for retrival efficacy.
    returns
    ----------
    flux distribution params
    """
    ### using m = -2.5log_10(F/zp)
    ### 10**(-(m/2.5))=F/zp
    ### F = 10**(-(m/2.5))/zp

    #print(F_u, F_l)
    mag_step = (mag_u - mag_l)/20.
    dist = np.zeros((20,2))
    for i in range(20):
        mag = mag_l+mag_step*i
        dist[i][0] = 1./20.
        dist[i][1] = 10**(-(mag/2.5))/zp

    return(dist)



def sample_flux(dist):
    """
    Ryan Cutter 2019


    Takes the distribtuion and selects a random value
    from that distribution 
 
    returns
    -----------
    integer flux value
    """


    LIST1=[] #List containing probabilities
    LIST2=[] #LIST containing flux_bins
    for i in range(len(dist)):
        LIST1.append(dist[i][0])
        LIST2.append(dist[i][1])
    flux_dist = LIST2[1] - LIST2[0]

    TMP = np.random.choice(len(dist), p=LIST1)
    random_flux = int(np.random.uniform(LIST2[TMP], LIST2[TMP]+flux_dist))

    return(random_flux)



def inject_fake_stars(image, n_stars, flux_dist, PSF_mod, write_cat, chop_x = 3, chop_y = 3):
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
    if write_cat == True:
        CAT = open('injection.cat', 'w')

    if PSF_mod == 1:
        PSF_dat = get_psf(image)

    if PSF_mod == 0:
        talk = [sex ,image,'-c',root_config+'/r_psf.sex' , '-CATALOG_NAME' , 'r.cat',
                '-PARAMETERS_NAME', root_config+'/r_psf.param', 
                '-FILTER_NAME', root_config+'/r_psf.conv'
                ]

        print('Making PSF cat')
        subprocess.call(talk, stderr=subprocess.DEVNULL)
        F, E, T = FWHM_shape('r.cat', image_dat, chop_x, chop_y) #get PSF stats
        X_b = np.linspace(0, X_s, chop_x+1) #define stat bounds
        Y_b = np.linspace(0, Y_s, chop_y+1)
        subprocess.call(['rm', 'r.cat'])

    for stars in range(n_stars):

        if isinstance(flux_dist, int) == True:
            flux = flux_dist
        else:
            flux = sample_flux(flux_dist)
 
        X_in = np.random.randint(0, X_s)
        Y_in = np.random.randint(0, Y_s)

        if PSF_mod == 0:
            for i in range(chop_x):
                for j in range(chop_y):
                    if X_b[i] <= X_in <= X_b[i+1] and Y_b[j] <= Y_in <= Y_b[j+1]:
                        kern_i = i
                        kern_j = j #j position in kern

            F_k = F[kern_i][kern_j]
            E_k = E[kern_i][kern_j]
            T_k = T[kern_i][kern_j]

            image_dat = inject_fake_source(flux, image_dat, PSF_mod, X_in, Y_in,
                                           FWHM = F_k, ELON = E_k, Theta = T_k, cut_size=80) #rounding is strange for cutout2D, stick to even cut_size

            if write_cat == True:
                CAT.write(str(X_in) + ' ' + str(Y_in) + ' ' + str(flux) +' \n') 



        elif PSF_mod == 1:
            image_dat = inject_fake_source(flux, image_dat, PSF_mod, X_in, Y_in,
                                            PSF_dat = PSF_dat, cut_size=80)
            if write_cat == True:
                CAT.write(str(X_in) + ' ' + str(Y_in) + ' ' + str(flux) +' \n')

        else:
            raise ValueError('Only two models \n PSF_mod = 0, basic bivariate Guassian \n PSF_mod = 1, PSFex model')

    return(image_dat)
	

def inject(image, n_stars, PSF_mod, out_name='inject.fits', flux_dist = 9000, chop_x = 3, chop_y = 3, overwrite = False , write_cat = True):
    """
    Ryan Cutter 2019

    Takes a fits file and injects n_stars into it
    Uses either: bivariate guassian (PSF_mod = 0)
                 psfex model        (PSF_mod = 1)
    """
    if PSF_mod == 0:
        image_dat = inject_fake_stars(image, n_stars, flux_dist, PSF_mod, write_cat, chop_x, chop_y)
    else:
        image_dat = inject_fake_stars(image, n_stars, flux_dist, PSF_mod, write_cat)

    fits.writeto(out_name, image_dat, overwrite = overwrite)

    return(None)


def write_mag(zp):
    """
    Ryan Cutter 2019


    Takes the written flux catalog and will
    convert flux to instrumental mag,using
    zp

    --------
    re-writes catalog 
    """


    new_cat = open('mags.cat', 'w')
    for lin in open('injection.cat', 'r'):
        tmp = lin.split()
        FL = float(tmp[2])
        MAG = -2.5*math.log10(FL/zp)
        new_cat.write(tmp[0]+' '+tmp[1]+' '+tmp[2]+' '+str(MAG)+'\n')

    print('Catalog format:')
    print('X, Y, Flux, Mag')
    
    subprocess.call(['rm','injection.cat'])
    return(None)



 






