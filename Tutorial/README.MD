### Basic usage

get_flux_dist(fits file name) 

mag_dist(bright_limit_magnitude, faint_limit_mgnitude, exposure time (seconds), pixel size(mm))

### Main Function

inject(fits file name, number of injected sources, psf model, out_name='inject.fits', flux_dist = 9000, chop_x = 3, chop_y = 3, overwrite = False, write_cat = True)

PSF_model = 0- Bivariate Guassian Model

PSF_model = 1- PSFex Model

chopx,chopy are slices for PSF_mod = 0. 

Overwrite to prevent overwriting out file

flux dist can either be a single value or a distribution found from either of the above


