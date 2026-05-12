'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph
'''

# Run on background thread, and refresh like every second?

import numpy as np
import sys

# Required as per DM-Script manual.
sys.argv.extend(['-a', ' '])
import matplotlib.pyplot as plt

import DigitalMicrograph.Py_Microscope as DM

import PyCTF
from PyCTF.image import ElectronImage
from PyCTF.image import import_ctf


# Define functions.
def _plot( fig, ax, x, y ):
	fig, ax = plt.subplots()
	ax.plot( x, y )
	return fig, ax


# Script starts here.
# Check on the main thread for using matplotlib in DM.
if ( DM.IsScriptOnMainThread() == False ):
	print( 'MatplotLib scripts are required to be run on the main thread.' )
	exit()

# Source front image. 
front_image = DM.GetFrontImage()
array = front_image.GetNumArray()
scale = DM.GetDimensionScale()
high_tension = DM.GetHighTension()

# Create ElectronImage class.
ctf = import_ctf( live_image, high_tension, scale )

# Subtract background and pre-processes.
ctf.remove_background( 8, 10 )
ctf.get_profiles()

# Extract the defocus.
ctf.measure_defocus()

# Plot the defocus and print it to the console.
# Or create as a 1D DM image to prevent weird crashes?
fig, ax = plt.subplots()

_plot( fig, ax, x, y )
plt.show()
# End of script.