'''
Defocus measurement.

A script to measure the defocus on the 2100F, using PyCTF in 
DigitalMicrograph

For use with DM, do make sure use numpy 1.23.5 and do not update.

Using dev branch with Numba and JIT.

ctrol+shift+q to kill script on background thread
'''

# Run on background thread, and refresh like every second?
# Use a thread to do the live FFT stuff, and a thread to display the results?

import numpy as np
import sys
import time

import DigitalMicrograph as DM

import pyCTF
from pyCTF.image import ElectronImage
from pyCTF.image import import_ctf
from pyCTF.fourier import Fourier
from pyCTF.profile import Profile


# Define functions.
def _plot( fig, ax, x, y ):
    fig, ax = plt.subplots()
    ax.plot( x, y )
    return fig, ax


# from export_insitu module
def _np_array_to_dm_image( input_array, **kwargs ):
    title = kwargs.get('title', None)
    dm_image = DM.CreateImage( input_array )
    if (title != None):
        dm_image.SetName( title )
    return dm_image


def _process_image( image, fft, prof ):
    # Get front image and extract numpy array.
    array = image.GetNumArray()
    
    # Fourier transform, subtract background.
    fft = Fourier.imfft( array )
    fft = Fourier.log_mod(fft)
    fft, _, _ = Fourier.remove_bckg( fft, 8, 10 )
    
    # Line profile.
    prof, _ = Profile.radial_profile( fft, len(fft[0])/2, len(fft[0])/2 )
    
    return fft, prof


def main_loop():
    # Get front image.
    front_image = DM.GetFrontImage()
    
    kv = 200
    scale = 1.0
    
    # 
    result_image = _np_array_to_dm_image( data, 'Processed image' )
    data = front_image.GetNumArray()
    
    # Init CTF object.
    # Linked to result_image via np array.
    ctf = import_ctf( data.copy(), kv, scale )
    ctf.image = Fourier.imfft( ctf.image )
    ctf.image = Fourier.logmod( ctf.image )
    ctf.remove_background( 10, 10 )
    
    # Do the defocus measurement.
    ctf.measure_defocus()
    
    profile = _np_array_to_dm_image( ctf.smoothed_profile, 'Profile' )
    
    # Show all the images.
    result_image.ShowImage()
    result_image.UpdateImage()
    
    profile.ShowImage()
    profile.UpdateImage()
    
    # Remove variables.
    del( ctf )
    del( data )
    return


# Script starts here.

main_loop()

# End of script.
