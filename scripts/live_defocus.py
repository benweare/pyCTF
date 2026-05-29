'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph

For use with DM, do make sure use numpy 1.23.5 and do not update.
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


# Script starts here.
# Check on the main thread for using matplotlib in DM.
if ( DM.IsScriptOnMainThread() == False ):
    print( 'MatplotLib scripts are required to be run on the main thread.' )
    exit()

# Source front image. 
front_image = DM.GetFrontImage()
array = front_image.GetNumArray()
scale = front_image.GetDimensionScale( 0 )
high_tension = DM.Py_Microscope().GetHighTension()

# Create ElectronImage class.
ctf = import_ctf( array, high_tension, scale )

# Fourier transform and convert to real type

# Subtract background and pre-processes.
#ctf.remove_background( 8, 10 )
#ctf.get_profiles()

# Extract the defocus.
#ctf.measure_defocus()

# Plot the defocus and print it to the console.
# Or create as a 1D DM image to prevent weird crashes?
# Py_LinePlotImageDisplay Class Reference
#fig, ax = plt.subplots()

#_plot( fig, ax, x, y )
#plt.show()


# display out
dm_img = _np_array_to_dm_image( ctf.image, title='ElectronImage' )
dm_img.ShowImage()

# End of script.
