'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph

For use with DM, do make sure use numpy 1.23.5 and do not update.

Must not use Numba and JIT as DM hangs.

'ctrl + shift + q' to kill scripts running on background thread.

Note: currently scales poorly as FFT-intensive.
Could use DM's FT to speed up?
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
            # Create an index that is incremented each time data is processed.
            self.i = 0
            # Get the original image and assign it to self.imgref.
            self.imgref = img
            
            # Get the data from the region within an ROI.
            self.roi = DM.GetROIFromID(self.find_ROI(self.imgref))
            val, val2, val3, val4 = self.roi.GetRectangle()
            self.data = self.imgref.GetNumArray()[int(val):int(val3),int(val2):int(val4)]
            # Get the shape and calibration of the original image.
            (input_sizex, input_sizey) = self.data.shape
            origin, x_scale, scale_unit =  self.imgref.GetDimensionCalibration(1, 0)
            self.i_scale = x_scale
            
            self.temp = Fourier.log_mod( self.data.copy() )
            self.temp[:], _, _ = Fourier.remove_bckg( self.temp.copy(), 8, 10 )
            
            self.prof, _ = Profile.radial_profile( self.temp.copy(), len(self.data[0])/2, len(self.data[0])/2 )
            self.dm_prof = self._np_array_to_dm_image( self.prof.copy(), title='RadialProfile' )
            self.prof = self.dm_prof.GetNumArray()
            
            # Set scale in DM image.
            self.dm_prof.SetDimensionScale( 0, self.i_scale )
            self.dm_prof.SetDimensionUnitString( 0, ('1/' + scale_unit) )
            
            
            # Show images and move them to right of source image.
            self.dm_prof.ShowImage()
            self.line_plot = (self.dm_prof.GetImageDisplay(0)).GetLinePlotImageDisplay()
            self.line_plot.SetContrastLimits( -0.2, 1.0)
            self._set_window_postion()
            
            DM.Py_ScriptObject.__init__(self)
            self.stop = 0
            
        except:
            print( traceback.format_exc() )
        return
    
    # Set the position of the new windows in DM.
    def _set_window_postion( self ):
        # Front image location
        image_doc = self.imgref.GetOrCreateImageDocument()
        doc_window = image_doc.GetWindow()
        size = doc_window.GetFrameSize()
        position = doc_window.GetFramePosition()
        # FFT location
        prof_doc = self.dm_prof.GetOrCreateImageDocument()
        prof_window = prof_doc.GetWindow()
        prof_window.SetFramePosition(size[0], position[1])
        prof_window.SetFrameSize( int(size[1]/2), int(size[1]/2) )
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
                #Get an (updated) ROI position
                val, val2, val3, val4 = self.roi.GetRectangle()
                
                #Get the data from the ROI area as a numpy array.
                self.data = self.imgref.GetNumArray()[int(val):int(val3),int(val2):int(val4)]
                self.temp = Fourier.log_mod( self.data.copy() )
                #self.temp[:], _, _ = Fourier.remove_bckg( self.temp.copy(), 8, 10 )
                self.prof[:], _ = Profile.radial_profile( self.temp.copy().astype(np.float64), len(self.data[0])/2, len(self.data[0])/2 )
                
                self.dm_prof.UpdateImage()
                self.line_plot.SetContrastLimits( -0.2, 1.0)
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
