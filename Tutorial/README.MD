### Main Function

inject(fits_file, number_of_stars, PSF_mod, grid = 0, out_name='inject.fits', flux_dist = 9000, chop_x = 3, chop_y = 3, overwrite = False , write_cat = True, x_p=-1, y_p=-1)

* PSF_model = 0- Bivariate Guassian Model
* PSF_model = 1- PSFex Model
* PSF_model = 2- Zernike Moments Model

* grid = 0 - random injections on input image
* grid = 1 - evenly spaced injections across input image
* grid = 2 - Inject in chose positions (x_p =[x1,x2,x3] y_p=[y1,y2,y3])
* grid = 3 - replicate input image with no noise
* grid = 4 - Pick your own postions no noise
* grid = 5 - Pick your own posotions fake noise 

chopx,chopy are slices for PSF_mod = 0. 

Overwrite to prevent overwriting out file

flux dist can either be a single value or a distribution found from either of the below


### Distribution Functions

get_flux_dist(fits file name) 

mag_dist(bright_limit_magnitude, faint_limit_mgnitude, exposure time (seconds), pixel scale)

**Grid=3 will copy input images fluxs and number of sources**
