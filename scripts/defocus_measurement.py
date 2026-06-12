'''

Defocus measurement script.

A script to measure the defocus from TEM images.

Author: E Weare
Location: nmRC
Contact: benjamin.weare1@nottingham.ac.uk-n0spam

Notes
-----
- For compatibility with DigitalMicrograph, use numpy v1.23.5
- Works on v3.62 on K3 server; and v3.60 on PC.
- DM prone to crash if using a package
 installed in editable mode using pip.
- Recommend running on background thread.

'''

import numpy as np

import DigitalMicrograph as DM

import pyCTF
from pyCTF.image import ElectronImage
from pyCTF.image import import_ctf
from pyCTF.fourier import Fourier


#from numba import jit, config
#config.DISABLE_JIT = False

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


# Calculate how to much to bin the FFT.
def _calc_bin_factor( pixel_size, target_nyquist ):
    nyquist = 1/pixel_size
    binning_factor = nyquist / target_nyquist
    return binning_factor

# Function to handle measuring the defocus using ctf object.
def _defocus_measure( ctf ):
    x = ctf.max_freq_inscribed
    # Get radial profiles.
    ctf.get_profiles( f_limits=[0, x], polynomial = 5 )
    # Get zeros.
    ctf.get_zeros()
    # Fit Cs and defocus using numpy method.
    m, c, cov = pyCTF.utils.fit( ctf.x_min, ctf.y_min, ctf.lamb )
    ctf.Cs, ctf.defocus = pyCTF.utils.calc_cs_and_defocus( m, c, ctf.lamb )
    return


# Set the position of the new windows in DM.
def _set_window_postion( ):
    # Front image location
    image_doc = self.imgref.GetOrCreateImageDocument()
    doc_window = image_doc.GetWindow()
    size = doc_window.GetFrameSize()
    position = doc_window.GetFramePosition()
    # FFT location
    fft_doc = self.dm_fft.GetOrCreateImageDocument()
    fft_window = fft_doc.GetWindow()
    fft_window.SetFramePosition(size[0], position[1])
    fft_window.SetFrameSize( int(size[1]/2), int(size[1]/2) )
    # Profile location
    size = fft_window.GetFrameSize()
    position = fft_window.GetFramePosition()
    prof_doc = self.dm_prof.GetOrCreateImageDocument()
    prof_window = prof_doc.GetWindow()
    prof_window.SetFramePosition(position[0], size[1])
    prof_window.SetFrameSize( size[0], size[0] )
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
    #_defocus_measure( ctf )
    
    print('\nFinished.')
    
    return ctf


# Script starts here.
print('\nStarting defocus measurement.')
front_image = DM.GetFrontImage()

ctf = main_loop( front_image )

print( 'Defocus = ' + str(ctf.defocus*1e-9) )

# Show all the images.
result_image = _np_array_to_dm_image( ctf.image, title='Cropped FFT of ' + front_image.GetName() )
result_image.SetDimensionScale( 0, ctf.scale )
result_image.SetDimensionScale( 1, ctf.scale )
result_image.SetDimensionUnitString( 0, '1/nm' )
result_image.SetDimensionUnitString( 1, '1/nm' )
result_image.ShowImage()

sprof = _np_array_to_dm_image( ctf.smoothed_profile, title='Smoothed Radial Profile' )
sprof.SetDimensionScale( 0, ctf.scale )
sprof.SetDimensionUnitString( 0, '1/nm' )

sprof.ShowImage()
# End of script