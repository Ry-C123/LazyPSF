[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2548023.svg)](https://doi.org/10.5281/zenodo.2548023)


# LazyPSF 

Lazy PSF is a python tool to inject fake stars into a photometric image.
~~~~~~~~~~~~~~~~~~~~
pip install lazypsf
~~~~~~~~~~~~~~~~~~~~

To get to grips with how to use lazypsf [See tutorial](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/lzypsf_tut.ipynb)

---

Will inject sources using either a bivariate Guassian:

![BGinjection](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/BGinj.png)

or PSFex

![PSFexinjection](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/PSinj.png)

---

Will model flux distribution from a given image and then return random fluxes from that same distribution:

![In_flux_dist](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/hist1.png)  ![Out_flux](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/hist2.png)


or a uniform distribution across mgnitudes

![Mag_dist](https://github.com/ryanc123/LazyPSF/blob/master/Tutorial/hist3.png)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ryan Cutter 
V1.1.0 (23/01/2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
