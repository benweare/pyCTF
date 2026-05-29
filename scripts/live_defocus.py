'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph

For use with DM, do make sure use numpy 1.23.5 and do not update.

Using dev branch with Numba and JIT.
'''

# Run on background thread, and refresh like every second?

import numpy as np
import sys

# Required as per DM-Script manual.
sys.argv.extend(['-a', ' '])
import matplotlib.pyplot as plt

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


def main_loop( image ):
    # Get front image and extract numpy array.
    array = image.GetNumArray()
    
    # Fourier transform, subtract background.
    fft = Fourier.imfft( array )
    fft = Fourier.log_mod(imfft)
    fft = Fourier.remove_bckg( fft, 8, 10 )
    
    # Line profile.
    prof = Profile.get_profile( fft, len(array[0])/2, len(image[0])/2 )
    
    return fft, prof


# Script starts here.
# Check on the main thread for using matplotlib in DM.
if ( DM.IsScriptOnMainThread() == False ):
    print( 'MatplotLib scripts are required to be run on the main thread.' )
    exit()

# Source front image. 
front_image = DM.GetFrontImage()
#scale = front_image.GetDimensionScale( 0 )
#high_tension = DM.Py_Microscope().GetHighTension()

main_loop( front_image )

dm_fft = _np_array_to_dm_image( fft, title='FFT' )
dm_prof = _np_array_to_dm_image( prof, title='RadialProfile' )
dm_fft.ShowImage()
dm_prof.ShowImage()

'''
array = front_image.GetNumArray()
scale = front_image.GetDimensionScale( 0 )
high_tension = DM.Py_Microscope().GetHighTension()

# Create ElectronImage class.
ctf = import_ctf( array, high_tension, scale )

dm_img = _np_array_to_dm_image( ctf.image, title='ElectronImage' )
dm_img.ShowImage()

# Does not work as uses Matplotlib, have to use DM show image path instead
#pyCTF.utils.show_image( ctf.image )

# Fourier transform and convert to real type
from pyCTF.fourier import Fourier
ctf.image = Fourier.imfft( ctf.image )
ctf.image = Fourier.log_mod( ctf.image )
ctf.remove_background(8, 10)

dm_fft = _np_array_to_dm_image( ctf.image, title='FFT' )
dm_fft.ShowImage()

# Subtract background and pre-processes.
from pyCTF.profile import Profile
prof, bins = Profile.radial_profile( ctf.image, ctf.centX, ctf.centY )

dm_prof = _np_array_to_dm_image( prof, title='SmoothedProfile' )
dm_prof.ShowImage()

# Extract the defocus.
#ctf.measure_defocus()

# Plot the defocus and print it to the console.
# Or create as a 1D DM image to prevent weird crashes?
# Py_LinePlotImageDisplay Class Reference
#fig, ax = plt.subplots()

#_plot( fig, ax, x, y )
#plt.show()


# display out
'''
# End of script.
