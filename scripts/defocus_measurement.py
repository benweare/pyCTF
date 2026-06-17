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


# Function to handle measuring the defocus using ctf object.
def _defocus_measure( ctf ):
    x = ctf.max_freq_inscribed
    # Get radial profiles.
    ctf.get_profiles( f_limits=[0, x], polynomial = 5 )
    # Get zeros.
    try:
        ctf.get_zeros()
        # Fit Cs and defocus using numpy method.
        m, c, cov = pyCTF.utils.fit( ctf.x_min, ctf.y_min, ctf.lamb )
        ctf.Cs, ctf.defocus = pyCTF.utils.calc_cs_and_defocus( m, c, ctf.lamb )
    except:
        ctf.Cs = 0
        ctf.defocus = 0
    return


# Set the position of the new windows in DM.
def _set_window_position( imgref, dm_fft, sprof ):
    # Front image location
    image_doc = imgref.GetOrCreateImageDocument()
    doc_window = image_doc.GetWindow()
    size = doc_window.GetFrameSize()
    position = doc_window.GetFramePosition()
    # FFT location
    fft_doc = dm_fft.GetOrCreateImageDocument()
    fft_window = fft_doc.GetWindow()
    fft_window.SetFramePosition(size[0], position[1])
    fft_window.SetFrameSize( int(size[1]/2), int(size[1]/2) )
    # Profile location
    size = fft_window.GetFrameSize()
    position = fft_window.GetFramePosition()
    prof_doc = sprof.GetOrCreateImageDocument()
    prof_window = prof_doc.GetWindow()
    prof_window.SetFramePosition(position[0], size[1])
    prof_window.SetFrameSize( size[0], size[0] )
    return


def find_ROI(image):
    '''
    Function to find an ROI placed on an image by the user, returning the ROI ID.
    If no ROI found, create a new one covering the entire image.
    '''
    imageDisplay = image.GetImageDisplay(0)
    numROIs = imageDisplay.CountROIs()
    id = None
    for n in range(numROIs):
        roi = imageDisplay.GetROI(n) 
        if roi.IsRectangle():
            roi.SetVolatile(False)
            roi.SetResizable(False)  
            id = roi.GetID()
            break
    if id is None:
        ## TO DO
        #If No ROI is found, create largest sqaure and center. 
        print("\nCreating square ROI.")
        data_shape = image.GetNumArray().shape
        print(data_shape)
        roi=DM.NewROI()
        # top, left, bottom, right
        roi.SetRectangle(0, ((data_shape[1]/2)-(data_shape[0]/2)), data_shape[0], ((data_shape[1]/2)+(data_shape[0]/2)))
        imageDisplay.AddROI(roi)
        roi.SetVolatile(False)
        roi.SetResizable(False)
        id = roi.GetID()
    return id

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
    
    
    
    roi = DM.GetROIFromID(find_ROI(front_image))
    val, val2, val3, val4 = roi.GetRectangle()
    data = front_image.GetNumArray()[int(val):int(val3),int(val2):int(val4)]
    
    # Scale of Fourier transform in 1/distance.
    i_scale = Fourier.calculate_scale( data, x_scale )
    
    # Init CTF object.
    # Linked to result_image via np array.
    
    # Fourier transform, crop, background subtract.
    
    binned_data = bin_array( data, 3.0, 1.0 )
    fft = Fourier.imfft( binned_data )
    #fft = Fourier.binned_imfft( data, x_scale, 2.0, 2.0, 'calc' )
    #fft = Fourier.crop( fft, len(fft[0])/4 )
    fft = Fourier.log_mod( fft )
    ctf = import_ctf( fft, kv, i_scale )
    del(fft)
    ctf.remove_background( 5, 5 )
    
    # Do the defocus measurement.
    _defocus_measure( ctf )
    
    print('\nFinished.')
    
    return ctf


# function to bin images.
def bin_array(data, binstep=2, binsize=2, func=np.sum):
    '''
    Function to bin arrays.
    '''
    # Function to bin array with numpy. Default is 2x binning.
    # See: https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    axes = [0, 1]
    data = np.array(data)
    dims = np.array(data.shape)
    for axis in axes:
        argdims = np.arange(data.ndim)
        argdims[0], argdims[axis]= argdims[axis], argdims[0]
        data = data.transpose(argdims)
        data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]/binstep)]
        data = np.array(data).transpose(argdims)
    return data


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

_set_window_position( front_image, result_image, sprof )
# End of script
