'''
Live defocus measurement.

A script to measure the defocus live on the 2100F, using PyCTF in 
DigitalMicrograph

For use with DM, do make sure use numpy 1.23.5 and do not update.

Using dev branch with Numba and JIT.

ctrol+shift+q to kill scripts running on background thread
'''

import numpy as np
import sys
import time
import traceback

# Required as per DM-Script manual.
#sys.argv.extend(['-a', ' '])
#import matplotlib.pyplot as plt

import DigitalMicrograph as DM

import pyCTF
from pyCTF.image import ElectronImage
from pyCTF.image import import_ctf
from pyCTF.fourier import Fourier
from pyCTF.profile import Profile


# From Ben Miller script
class imageListener( DM.Py_ScriptObject ):
    '''
    Image listener class.
    
    Based on script by Ben Miller for live data processing.
    '''
    
    
    # Constructor.
    def __init__(self,img):
        try:
            #Create an index that is incremented each time data is processed.
            self.i = 0
            #get the original image and assign it to self.imgref
            self.imgref = img
            
            #Get the data from the region within an ROI
            self.roi = DM.GetROIFromID(self.find_ROI(self.imgref))
            val, val2, val3, val4 = self.roi.GetRectangle()
            self.data = self.imgref.GetNumArray()[int(val):int(val3),int(val2):int(val4)]
            #get the shape and calibration of the original image
            (input_sizex, input_sizey) = self.data.shape
            #origin, x_scale, scale_unit =  self.imgref.GetDimensionCalibration(1, 0)
            #if scale_unit == b'\xb5m': scale_unit = 'um' #scale unit of microns causes problems for python in DM
            
            #Create a new image to contain the results of processing.
            self.result_image = DM.CreateImage(self.ROI_process(self.data))
            #Set the calibration based on the original data
            self.result_image.SetDimensionCalibration(0,origin,x_scale,scale_unit,0)
            self.result_image.SetDimensionCalibration(1,origin,x_scale,scale_unit,0)
            #Get the numpy array of the result image so I can directly change the data values later
            self.result_data=self.result_image.GetNumArray()
            
            #Set the image name which will be displayed in the image window's title bar
            self.result_image.SetName("Extract of " + img.GetName())
            self.result_data=self.result_image.GetNumArray()
            
            # CTF variables - causing some issue with the PyScriptObject class?
            fft, prof = self._process_image( self.result_data )
            # is it this part?
            self.dm_fft = self._np_array_to_dm_image( fft, title='FFT' )
            self.dm_prof = self._np_array_to_dm_image( prof, title='RadialProfile' )
            
            # Get reference for np arrays.
            self.fft = dm_fft.GetNumArray()
            self.prof = dm_prof.GetNumArray()
            
            # Show images.
            self.result_image.ShowImage()
            self.dm_fft.ShowImage()
            self.dm_prof.ShowImage()
            
            DM.Py_ScriptObject.__init__(self)
            self.stop = 0
        except:
            print( traceback.format_exc() )
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
            #If No ROI is found, create one that covers the whole image. 
            print("\nRectangular ROI not found... using whole image")
            data_shape = image.GetNumArray().shape
            roi=DM.NewROI()
            roi.SetRectangle(0, 0, data_shape[0], data_shape[1])
            imageDisplay.AddROI(roi)
            roi.SetVolatile(False)
            roi.SetResizable(False)
            id = roi.GetID()
        return id
        
        
        # Function run each time updates.
    def ROI_process(self, image_data):
        '''
        Function to be called every time data is updated.
        '''
        image_data.copy()
        fft, prof = self._process_data( image_data )
        return image_data, fft, prof
    
    
    def _np_array_to_dm_image( self, input_array, **kwargs ):
        title = kwargs.get('title', None)
        dm_image = DM.CreateImage( input_array )
        if (title != None):
            dm_image.SetName( title )
        return dm_image
    
    
def _process_image( self, array ):
    # Fourier transform, subtract background.
    fft = Fourier.imfft( array )
    fft = Fourier.log_mod(fft)
    fft, _, _ = Fourier.remove_bckg( fft, 8, 10 )
    # Line profile.
    prof, _ = Profile.radial_profile( fft, len(fft[0])/2, len(fft[0])/2 )
    return fft, prof
    
    
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
                
                #Process the data and place in the result arrays.
                self.result_data[:], self.fft[:], self.prof[:] = self.ROI_process( self.data )
                
                #Update the image displays.
                self.result_image.UpdateImage()
                self.dm_fft.UpdateImage()
                self.dm_prof.UpdateImage()
                
                #Increment an index each time data is processed.
                self.i = self.i+1
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