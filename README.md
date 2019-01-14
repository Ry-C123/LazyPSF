# LazyPSF 

Lazy PSF is a tool to inject fake stars into a photometric image.

It uses Sextractor to obtain PSF features an injects sources with the found features. You can also get this code to spit out a PSF kernel for a given image.

For now:
Uses a Guassian bivariate distribution to model the PSF across the field. Accounts for rotation of the PSF

Main issue:
The injected source can have patches in it due to the rotation.

Future release:
Will have a more advanced option of injection a kernel extraction using PSFex

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ryan Cutter 
V0.1.0 (14/01/2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
