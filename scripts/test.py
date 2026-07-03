'''
Standard tests for PyCTF.
'''

# Timing variables for testing
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# t1 = time.perf_counter()
# t2 = time.perf_counter()
# diff = t2 - 1

# Test importing package.
import pyCTF


from numba import jit, config
if config.DISABLE_JIT == True:
	print('Numba JIT disabled.')
else:
	print('Numba JIT enabled.')

# Test 1D simulations
def _test_1D_sim():
	print('Testing 1D simulations.')

	from pyCTF.simulation import CTFSimulation1D
	import matplotlib.pyplot as plt

	CTF = CTFSimulation1D( 5.0, int(2000), 200, -500,
		delta_current=0.0001,
		current=5.0,
		beta=0.1,#mrad
		delta_E=3.0,
		delta_voltage=0.0001,
		aperture=2.5,
		mode=0)

	fig, ax = CTF.plot_ctf()
	#ax.set_prop_cycle(color=['red', 'green', 'blue'])
	ax.set_ylim([0,None])

	CTF.show_all()

	CTF.print_aberrations()
	return

# Test 2D simulations
def _test_2D_sim():
	print('Testing 2D simulations.')

	from pyCTF.simulation import CTFSimulation2D
	image_size = int(256) # Side length of image in pixels.
	acc_voltage = 200 # Accelerating voltage in kV.
	max_frequency = 6 # Maximum spatial frequency in nm-1.
	defocus = -300 # Defocus in nm.

	CTF2D = CTFSimulation2D( max_frequency,
	                         image_size,
	                         acc_voltage,
	                         defocus,
	                         aperture=3.0,
	                         C12a = 400,
	                         C12b = 0,
	                         phi = np.deg2rad(45),
	                         mode=0 )
	CTF2D.defocus=1000e-9
	CTF2D.update()
	CTF2D.plot_ctf()
	CTF2D.show_all()
	CTF2D.print_aberrations()
	print(CTF2D.scale)
	return


# Test defocus measurement.
def _test_defocus( filepath ):
	print('Testing defocus measurements.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image
	from pyCTF.utils import normalise_data_range

	
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )

	CTF.remove_background( 8, 4 )

	from pyCTF.image import measure_defocus
	from pyCTF.image import print_Cs_results
	measure_defocus( CTF,
	                 polynomial=20,# Polynomial for Savitsky-Golay smoothing.
	                 window=3,# Window size for Savitsky-Golay smoothing.
	                 f_limits=[0.1,3.0],# Range of freqeuncy to fit.
	                     underfocus=True,# False for overfocus.
	                     xlim=[0.75,3.0],# Exclude all minima outside this range.
	                     start=2 )# First index for fitting.
	print_Cs_results( CTF, verbose=True )

	plt.show()
	return

def _test_background_subtraction( filepath ):
	print('Testing background subtraction.')
	from PIL import Image
	from pyCTF.utils import show_image
	from pyCTF.image import ElectronImage

	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )

	CTF.remove_background( 8, 4 )

	show_image( CTF.image, scale=CTF.scale )
	CTF.plot_background()

	return

# Testing astig functions.
def _test_astig( filepath ):
	print('Testing astigmatism measurements.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image

	# Import the CTF and remove the background.
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.remove_background(10, 10)

	from pyCTF.astig import astig_angle
	from pyCTF.astig import Astig
	# Measure the angle of twofold astigmatism.
	astig_angle( CTF )
	Astig.plot_angles( CTF )
	print( CTF.amax )
	return


# Testing chromatic aberration functions.
def _test_chromatic():
	print('Testing chromatic aberrations measurements.')
	from pyCTF.chromatic import chromaticAberration

	voltage = 200
	voltage_data = np.array([199.70, 199.75, 199.80, 199.85, 199.90, 199.95, 200])
	defocus_data =   np.array([ -714.09, -433.04, -197.66, 0, 240.20, 458.89, 688.94 ])*1e-9

	Cc = chromaticAberration( voltage,# Primary accelerating voltage in kV.
	                          voltage_data,# Series of accelerating voltages in kV.
	                          defocus_data)# Series of defocuses, in m.
	fit_method = 'lmfit'
	Cc.fit( method=fit_method )
	Cc.plot_figure( method=fit_method )
	Cc.print_results( method=fit_method )
	return

def _test_other( filepath ):
	print('Testing overlaying images.')
	from pyCTF import image
	from pyCTF.utils import composite_image, show_image, normalise_data_range
	from pyCTF.simulation import CTFSimulation2D

	# First, import the experimental CTF.
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.remove_background(10, 10)
	# Then, simulate a 2D CTF.
	image_size = CTF.width
	acc_voltage = 200
	max_frequency = CTF.max_freq_inscribed*2
	defocus = -484
	simCTF = CTFSimulation2D(max_frequency, image_size, acc_voltage, defocus)
	simCTF.Cs=1.3e-3
	simCTF.update()

	# Then create the overlay.
	composite = composite_image( CTF.image,# Base image.
	                 simCTF.square_CTF,# Image to overlay.
	                 int(CTF.width/2) )# Size of output.
	show_image( composite, scale=CTF.scale )
	return

def _test_fourier( filepath ):
	print('Testing Fourier functions.')
	from pyCTF.fourier import Fourier
	from pyCTF.utils import show_image

	image =  np.array( Image.open( filepath ))
	FFT = Fourier.imfft( image )
	FFT = Fourier.crop( FFT, 600 )
	FFT = Fourier.log_mod( FFT )

	show_image( FFT, scale=0.14782, length=5 )
	return


def _test_TFS( filepath ):
	print('Testing TFS.')

	from pyCTF.fourier import Fourier
	from pyCTF.utils import show_image

	stack =  Fourier.import_stack( filepath )
	FFT, prof = Fourier.through_focus( stack,
	                                   width=60,
	                                   r1=5,
	                                   r2=5,
	                                   verbose=True)

	fig, ax = plt.subplots( 1,2, figsize=(8,8) )
	ax[0].matshow(FFT[:,:,18], aspect=1)
	ax[1].matshow( np.rot90(prof[1:,:]) )
	ax[1].set_ylabel('slice')
	ax[1].set_xlabel('Frequency / a.u.')
	return


def _test_custom( filepath ):
	print('Custom tests.')
	from pyCTF.fourier import Fourier
	from pyCTF.utils import show_image

	#image =  np.array( Image.open( filepath ))
	#FFT = Fourier.binned_imfft( image, 0.098486, 2.5, 'calc' )
	#FFT = Fourier.imfft( image )
	#FFT = Fourier.crop( FFT, 600 )
	#FFT = Fourier.log_mod( FFT )

	#image = np.ones((100,100),dtype=np.float64)
	image = np.random.choice(np.arange(100, dtype=np.int32), size=(100, 100))
	image = pyCTF.utils.normalise_data_range(image)
	image[49,49]=10
	bf = 4

	FFT = pyCTF.utils.bin_array( image, bf, bf)
	show_image( FFT, scale=0, length=2 )
	return

def _test_profiles( filepath ):
	print('Testing line profiles.')

	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image
	from pyCTF.utils import normalise_data_range

	
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	CTF.get_profiles(f_limits=[0, 3.0])

	CTF.plot_profiles()
	
	plt.show()
	return

# Script starts here.
print('Starting tests.')

path = 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\'

#_test_1D_sim()
#_test_2D_sim()
_test_defocus( path + 'example_CTF.tif' )
#_test_profiles( path + 'example_CTF.tif' )
#_test_background_subtraction( path+'example_CTF.tif' )
#_test_astig( 'assets\\example_astigmatism.tif' )
#_test_chromatic()
#_test_other( path+'example_CTF.tif' )
#_test_fourier('assets\\example_image.tif')
#_test_TFS( 'assets\\example_TFS.tif' )
#_test_custom( 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\test_img.tif' )

print('Tests finished.')

plt.show()