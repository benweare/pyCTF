'''

Defocus measurement script.

A script to measure the defocus from TEM images.

Author: E Weare
Location: nmRC
Contact: benjamin.weare1@nottingham.ac.uk-n0spam

Notes
-----
For compatibility with DigitalMicrograph, use numpy v1.23.5

'''


import numpy as np

import DigitalMicrograph as DM

import pyCTF
from pyCTF.image import ElectronImage
from pyCTF.image import import_ctf

from pyCTF.fourier import Fourier


from numba import jit, config
config.DISABLE_JIT = True

# Define functions.

# From GiveMeED export_insitu module.
# Wrapper for to make DM images from np array.
def _np_array_to_dm_image( input_array, **kwargs ):
    title = kwargs.get('title', None)
    dm_image = DM.CreateImage( input_array )
    if (title != None):
        dm_image.SetName( title )
    return dm_image


def _calc_scale( image, scale ):
    iscale = 1/( len(image[0]) * scale )
    return iscale


# Function to handle measuring the defocus using ctf object.
def _measure_defocus( ctf ):
    # Get radial profiles.
    ctf.get_profiles( f_limits=[0,ctf.max_freq_inscribed], polynomial = 5 )
    # Get zeros.
    ctf.get_zeros()
    # Fit Cs and defocus using numpy method.
    m, c, cov = pyCTF.utils.fit( ctf.x_min, ctf.y_min, ctf.lamb )
    ctf.Cs, ctf.defocus = pyCTF.utils.calc_cs_and_defocus( m, c, ctf.lamb )
    return


# TO DO: add units to output FFT DM image.
def main_loop( front_image ):
    '''
    Main processing function.
    
    Takes front image, converts to CTF, and takes radial profiles to measure
    the defocus.
    
    
    Returns
    -------
    ctf : object
        pyCTF ctf object containing processed data.
    
    
    Notes
    -----
    Uses pyCTF module for data processing, and DM module for interacting
    with DigitalMicrograph images.
    '''
    
    print('\nStarting.')
    # Get front image.
    origin, x_scale, scale_unit =  front_image.GetDimensionCalibration(1, 0)
    kv = 200
    

    data = front_image.GetNumArray()
    # Scale of Fourier transform in 1/distance.
    i_scale = _calc_scale( data, x_scale )
    
    
    # Init CTF object.
    # Linked to result_image via np array.
    
    # Fourier transform, crop, background subtract.
    fft = Fourier.imfft( data )
    fft = Fourier.crop( fft, len(fft[0])/4 )
    fft = Fourier.log_mod( fft )
    ctf = import_ctf( fft, kv, i_scale )
    del(fft)
    ctf.remove_background( 5, 5 )
    
    # Do the defocus measurement.
    _measure_defocus( ctf )
    
    print('\nFinished.')
    
    return ctf


# Script starts here.
front_image = DM.GetFrontImage()

ctf = main_loop( front_image )

print( ctf.defocus )

# Show all the images.
result_image = _np_array_to_dm_image( ctf.image, title='Cropped FFT of ' + front_image.GetName() )
result_image.ShowImage()

sprof = _np_array_to_dm_image( ctf.smoothed_profile, title='Smoothed Radial Profile' )

sprof.ShowImage()

# End of script.
