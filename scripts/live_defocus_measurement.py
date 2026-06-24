'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph.

For use with DM, do make sure to use Numpy 1.23.5 and do not update.

'ctrl + shift + q' to kill scripts running on background thread.

Notes:
- Runs well up to 1K resolution, 10 fps & 20 fps.
- Noticable slowdown at 2K, 10 fps.
- Main limit is number of pixels in the ROI.
- Taking an image will end the script.
- Update so the calibration changes if the mag changes.
'''

import numpy as np
import sys
import time
import traceback

from time import sleep

import DigitalMicrograph as DM

import pyCTF
from pyCTF.image import ElectronImage
from pyCTF.image import import_ctf
from pyCTF.fourier import Fourier
from pyCTF.profile import Profile
from pyCTF.zeros import Zeros

from pyCTF.utils import kv_to_lamb

#from numba import jit, config
#config.DISABLE_JIT = False


class imageListener( DM.Py_ScriptObject ):
    '''
    Image listener class.
    
    Based on script by Ben Miller for live data processing.
    '''
    
    
    # Constructor.
    def __init__( self, img ):
        try:
            # Get the original image and assign it to self.imgref.
            self.imgref = img
            
            # Get the data from the region within an ROI.
            self.roi = DM.GetROIFromID(self.find_ROI(self.imgref))
            val, val2, val3, val4 = self.roi.GetRectangle()
            self.data = self.imgref.GetNumArray()[int(val):int(val3),int(val2):int(val4)]

            # Get the shape and calibration of the original image.
            (input_sizex, input_sizey) = self.data.shape
            origin, x_scale, scale_unit =  self.imgref.GetDimensionCalibration(1, 0)
            
            # Calculate the binning factor for the target frequency.
            #self.binning_factor = Fourier.calculate_bin_factor( len(self.data[0]), x_scale, 3.0 )
            
            # Create the Fourier transform, then crop to size and make a DM image.
            self.fft = Fourier.log_mod( Fourier.imfft( self.data ))
            self.fft = Fourier.crop( self.fft.copy(), len(self.fft[0])/2 )
            self.fft = Fourier.log_mod( self.fft )
            
            # Use center to mask out DC frequencies from live transform.
            self.fft_center = int( len(self.fft[0])/2 )

            # Set up CTF object and precompute iradius for background subtraction.
            self.kV = DM.Py_Microscope().GetHighTension()/1000
            i_scale = Fourier.calculate_scale( self.data, x_scale )
            self.lamb = kv_to_lamb( self.kV )
            self.iradius, _ = pyCTF.utils.find_iradius_itheta( self.fft.copy(), 1 )

            # Precompute image masks for live background subtraction.
            self.lf_mask = self._create_masks( 8, self.iradius )
            self.hf_mask = self._create_masks( 10, self.iradius )

            # Initial background removal.
            self.fft[:] = self._remove_bckg( self.fft.copy() )
            self.dm_fft = self._np_array_to_dm_image( self.fft, title='FFT' )
            # Make a ref to the image data so can update the image live.
            self.fft = self.dm_fft.GetNumArray()
            
            # Calculate the frequency to crop the radial profile to.
            self.max_freq= int( 2.0/i_scale )
            
            # Create the radial profile for the Fourier transform.
            self.r, self.nr = self._profile_precompute( self.fft, len(self.fft[0])/2, len(self.fft[0])/4 )
            self.prof = self._fast_profile( self.fft )
            self.prof = self.prof[:self.max_freq]
            self.dm_prof = self._np_array_to_dm_image( self.prof.copy(), title='RadialProfile' )
            self.prof = self.dm_prof.GetNumArray()
            
            # Make a buffer to store the last 10 frames.
            self.buffer = np.zeros( (self.prof.shape[0], 10) )
            self.i = 0
            print(self.buffer.shape)
            print(self.prof.shape)
            
            # Set scale of the DigitalMicrograph images.
            self.dm_fft.SetDimensionScale( 0, i_scale )
            self.dm_fft.SetDimensionScale( 1, i_scale )
            self.dm_prof.SetDimensionScale( 0, i_scale )
            self.dm_prof.SetDimensionUnitString( 0, ('1/' + scale_unit) )
            self.dm_fft.SetDimensionUnitString( 0, ('1/' + scale_unit) )
            self.dm_fft.SetDimensionUnitString( 1, ('1/' + scale_unit) )
            
            # Show the images and move them to right of source image.
            self.dm_fft.ShowImage()
            self.dm_prof.ShowImage()
            self.prof_img_display = self.dm_prof.GetImageDisplay(0)
            self.line_plot = self.prof_img_display.GetLinePlotImageDisplay()
            #self.line_plot.SetFilled( False )
            #self.line_plot.SetSliceComponentColor( 0, 0, 0, 0 ,0 )
            self.line_plot.SetSliceDrawingStyle( 0, 3 )
            self.line_plot.SetGridColor( 99, 99, 99 )
            self.line_plot.SetDoAutoSurvey( False, False )
            self.line_plot.SetContrastLimits( -0.5, 0.5)
            self._set_window_postion()
            
            DM.Py_ScriptObject.__init__(self)
            self.stop = 0
            
        except:
            print( traceback.format_exc() )
        return


    def _profile_precompute( self, data, centX, centY ):
        y, x = np.indices( data.shape )
        r = np.sqrt( np.square(x - centX) + np.square(y - centY) )
        r = r.astype( np.int64 )
        nr = np.bincount( r.ravel() )
        return r, nr

    # Uses precompute to speed up.
    def _fast_profile( self, data ):
        tbin = np.bincount( self.r.ravel(), data.ravel() )
        radialprofile = tbin / self.nr
        return radialprofile

    # Remove background via Fourier method but with values precomputed.
    def _remove_bckg( self, image ):
        # low frequency
        imfft = np.fft.fft2( image )
        imfft = np.fft.fftshift( imfft )
        imfft = imfft * self.lf_mask
        imfft = np.fft.ifft2( imfft )
        LF_bkg = np.abs( imfft )
        image = image - np.abs( imfft )
        # high frequency
        imfft = np.abs( image ) #natural log
        imfft = np.fft.fft2( imfft)
        imfft = np.fft.fftshift( imfft )
        imfft = imfft * self.hf_mask
        imfft = np.exp( np.abs(np.fft.ifft2( imfft )) )
        image = image / imfft
        return image

    '''
    # Function to handle measuring the defocus using ctf object.
    def _measure_defocus( self, data ):
        try:
            minima = scipy.signal.find_peaks( -data )
            # Filter out bad points.
            #minima = np.array([x for x in minima if freq[x] <= xlim[1]])
            #minima = np.array([x for x in minima if freq[x] >= xlim[0]])
            #minima = np.array([x for x in minima if prof[x] <= ylim[1]])
            #minima = np.array([x for x in minima if prof[x] >= ylim[0]])
            x_min = ( freq[ minima ] )**2
            # Fit Cs and defocus using numpy method.
            m, c, cov = pyCTF.utils.fit( x_min, y_min, self.lamb )
            self.Cs, self.defocus = pyCTF.utils.calc_cs_and_defocus( m, c, self.lamb )
            print('\r Defocus = ' + str(self.ctf.defocus) )
        except:
            print('\r Could not fit defocus.')
            ctf.Cs = 0
            ctf.defocus = 0
        return
    '''



    # Compute masks for backfround subtraction.
    def _create_masks( self, start_radius, iradius ):
        n = range(0, np.size(iradius,0))
        m = range(0, np.size(iradius,1))
        mask = np.ones( self.fft.shape )
        for i in n:
            for j in m:
                if iradius[i,j] >= start_radius:
                    mask[i,j] = 0
        return mask
    

    # Set the position of the new windows in DM.
    def _set_window_postion( self ):
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
        fft_disp = self.dm_fft.GetImageDisplay(0)
        try:
            fft_disp.ImageDisplaySetInputColorTable( 'Viridis' )
        except:
            print('\nCould not set colour table.')
        # Profile location
        size = fft_window.GetFrameSize()
        position = fft_window.GetFramePosition()
        prof_doc = self.dm_prof.GetOrCreateImageDocument()
        prof_window = prof_doc.GetWindow()
        prof_window.SetFramePosition(position[0], size[1])
        prof_window.SetFrameSize( size[0], size[0] )
        return
    
    
    def find_ROI(self, image):
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
            # x1, y1, x2, y3
            roi.SetRectangle(0, ((data_shape[1]/2)-(data_shape[0]/2)), data_shape[0], ((data_shape[1]/2)+(data_shape[0]/2)))
            imageDisplay.AddROI(roi)
            roi.SetVolatile(False)
            roi.SetResizable(False)
            id = roi.GetID()
        return id
    
    
    
    def _np_array_to_dm_image( self, input_array, **kwargs ):
        title = kwargs.get('title', None)
        dm_image = DM.CreateImage( input_array )
        if (title != None):
            dm_image.SetName( title )
        return dm_image
    
    
    
    def HandleDataChangedEvent(self, flags, image):
        '''
        This function is run each time the image changes.
        '''
        try:
            if not self.stop:
                # Get an updated ROI position.
                val, val2, val3, val4 = self.roi.GetRectangle()
                
                # Get the data from the ROI area as a numpy array.
                self.data = self.imgref.GetNumArray()[int(val):int(val3),int(val2):int(val4)]

                # Store the FFT before it is cropped to the right size.
                temp = Fourier.log_mod( Fourier.imfft( self.data ))
                # Process the Fourier transform and line profile.
                self.fft[:] = Fourier.crop( temp, len(self.fft[0]) )
                self.fft[:] = self._remove_bckg( self.fft )
                self.fft[ self.fft_center, self.fft_center] = 0
                temp = self._fast_profile( self.fft )*2
                self.prof[:] = temp[:self.max_freq]
                #baseline = Profile.remove_baseline( self.prof )
                self.prof[:] = Profile.smooth_profile( self.prof, 10, 5 )
                
                # Update the live images.
                self.dm_fft.UpdateImage()
                self.dm_prof.UpdateImage()
                
                # Update the buffer.
                self.buffer[:, self.i] = self.prof[:]
                if self.i == 9:
                    self.i = 0
                else:
                    self.i = self.i+1
                
                #Increment an index each time data is processed.
                
        except:
            print(traceback.format_exc())
        return
    
    
    # Destructor.
    def __del__( self ):
        print( 'Listener Deleted' )
        DM.Py_ScriptObject.__del__(self)
        return
    
    
    #Function to end processing by deleting or unregistering listener
    def RemoveListeners(self):
        try: 
            if not self.stop:
                self.stop = 1
                DM.DoEvents()
                listener.UnregisterAllListeners()
                print("Live Processing Script Ended")
        except:
            print(traceback.format_exc())
        return
    
    
    #Remove listeners if source image window is closed
    def HandleWindowClosedEvent(self, event_flags, window):
        print("Window Closed")
        self.RemoveListeners()
        return
    
    
    #Remove listeners if the ROI is deleted
    def HandleROIRemovedEvent(self, img_disp_event_flags, img_disp, roi_change_flag, roi_disp_change_flags, roi):
        print("ROI Removed")
        self.RemoveListeners()
        return
    
    
    #Check if running on the main thread for using matplotlib in DM.
    def _check_thread( self ):
        if ( DM.IsScriptOnMainThread() == False ):
            print( 'MatplotLib scripts are required to be run on the main thread.' )
            exit()
        return


# Script starts here.
front_image = DM.GetFrontImage()
image_doc = DM.GetFrontImageDocument()
im_doc_win = image_doc.GetWindow()
image_display = front_image.GetImageDisplay(0)

listener = imageListener( front_image )

WindowClosedListenerID = listener.WindowHandleWindowClosedEvent(im_doc_win, 'pythonplugin')
ROIRemovedListenerID = listener.ImageDisplayHandleROIRemovedEvent(image_display,'pythonplugin')
DataChangedListenerID = listener.ImageHandleDataChangedEvent(front_image, 'pythonplugin')

# End of script.
