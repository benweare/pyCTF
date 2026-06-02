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

# Required as per DM-Script manual.
#sys.argv.extend(['-a', ' '])
#import matplotlib.pyplot as plt

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

def _update_display( dm_fft, dm_prof ):
    #array = dm_fft.GetNumArray()
    #array = array * 0
    #imgDoc = dm_fft.# get the image document, delete the old image, add the new one?
    return

def main_loop():
    front_image = DM.GetFrontImage()
    fft = None
    prof = None
    fft, prof = _process_image( front_image, fft, prof )
    dm_fft = _np_array_to_dm_image( fft, title='FFT' )
    dm_prof = _np_array_to_dm_image( prof, title='RadialProfile' )
    dm_fft.ShowImage()
    dm_prof.ShowImage()
    # Get object reference for numpy array (DM is silly).
    fft = dm_fft.GetNumArray()
    prof = dm_prof.GetNumArray()
    # Do the defocus measurement.
    _destructor()
    return

def _destructor():
    # Function to close thread and delete all variables when script ends.
    del( fft )
    del( prof )
    return

# Script starts here.
# Check on the main thread for using matplotlib in DM.
#if ( DM.IsScriptOnMainThread() == False ):
#    print( 'MatplotLib scripts are required to be run on the main thread.' )
#    exit()

main_loop()

# End of script.
