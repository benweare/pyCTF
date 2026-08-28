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


#from numba import jit, config
#if config.DISABLE_JIT == True:
#	print('Numba JIT disabled.')
#else:
#	print('Numba JIT enabled.')

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
	fig, ax = Cc.plot_figure( method=fit_method )
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
	CTF.image = normalise_data_range(CTF.image)
	CTF.image = pyCTF.utils._mask_dc_frequency( CTF.image, 0.4, 3)
	CTF.get_profiles(f_limits=[0, 3.0])

	#CTF.plot_profiles()
	fig, ax = plt.subplots(1)
	ax.plot( CTF.frequency, CTF.radial_profile )
	ax.set_box_aspect(1)
	ax.set_ylabel('Intensity / a.u.', fontsize = 16)
	ax.set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
	ax.set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
	ax.set_xticks([])
	ax.legend()
	plt.show()
	return

def _test_show_image( filepath ):
	from pyCTF.image import ElectronImage
	from pyCTF.utils import show_image
	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127 )
	show_image( CTF.image, scale=CTF.scale, units='nm', cmap='Greys' )
	return

def _test_astig_measure(filepath):
	from pyCTF.image import ElectronImage
	from pyCTF.astig import Astig
	from pyCTF.utils import normalise_data_range

	CTF = pyCTF.image.import_ctf( np.array( Image.open( filepath )), 
												200, 
												0.0066127*2)
	CTF.remove_background( 15, 10 )

	#from pyCTF.astig import astig_angle
	#astig_angle( CTF )
	#Astig.plot_angles( CTF )
	#print( CTF.amax )

	#from pyCTF.utils import show_image
	#show_image(CTF.image)

	from pyCTF.simulation import CTFSimulation2D
	image_size = CTF.image.shape[0] # Side length of image in pixels.
	#CTF.image[int(CTF.image.shape[0]/2), int(CTF.image.shape[0]/2)] = 0

	CTF2D = CTFSimulation2D( CTF.image.shape[0]*0.0066127*2,
	                         image_size,#size
	                         200,#kv
	                         -650)# defocus
	CTF2D.phi = np.deg2rad(40.3)
	CTF2D.update()

	slices = 15

	vals, a, polar_list = Astig.magnitude_measure( normalise_data_range(CTF.image),
													slices,
													100,
													CTF2D )
	Astig.plot_results( CTF, vals, slices, a, CTF2D )
	return


def _TFS( filepath):
	# Function to process the through-focus series into a stack of 2D
	# Fourier transforms.
	from pyCTF.fourier import Fourier
	print('Processing TFS.')

	# Define radii for background subtraction.
	r1 = 4
	r2 = 4

	print('Importing through-focus series.')
	stack =  Fourier.import_stack( filepath )
	print('Performing FFT.')
	FFT = Fourier.fft_stack( stack )
	FFT = Fourier.log_mod( FFT )
	print('Removing background.')
	FFT = Fourier.remove_bckg_stack( FFT, r1, r2 )
	print('Radial profile...')
	prof = Fourier.profile_fft_stack( FFT )

	fig, ax = plt.subplots()
	ax.imshow(prof)

	np.save('TFS', prof)

	return

def  _test_zer( filepath ):
	arr = np.load( filepath )
	arr = np.rot90(arr)
	arr = arr[:, 1:50]

	import skimage
	from skimage.filters import gaussian
	arr = gaussian(arr, sigma=1)

	#from pyCTF.utils import normalise_data_range
	#arr = normalise_data_range(arr)

	#fig, ax = plt.subplots()
	#ax.imshow(arr)

	freq = np.linspace(0, arr.shape[1])

	output = np.zeros([arr.shape[0], 8])

	from pyCTF.zeros import Zeros

	for n in range(arr.shape[0]):
		minima, _ = Zeros.calc_zeros( arr[n, :] )
		minima = Zeros.filter_zeros(minima, arr[n, :], freq, [10,40], [None,0.4] )
		for m in range(minima.shape[0]):
			output[n, m] = minima[m]

	output[output == 0] = np.nan
	'''
		try:
			ax.plot(freq[:][minima[0]], arr[n,:][minima[0]]+n, 'x', color='k', label='n=|1|')
			ax.plot(freq[:][minima[1]], arr[n,:][minima[1]]+n, 'x', color='y', label='n=|2|')
			ax.plot(freq[:][minima[2]], arr[n,:][minima[2]]+n, 'x', color='m', label='n=|3|')
			ax.plot(freq[:][minima[3]], arr[n,:][minima[3]]+n, 'x', color='c', label='n=|4|')
		except:
			pass
	#ax.legend()
	'''
	
	fig, ax = plt.subplots()
	ax.imshow(arr)

	ax.plot( output[:,0], range(0, 200),  'x', color='k', label='n=|1|')
	ax.plot( output[:,1], range(0, 200), 'x', color='y', label='n=|2|')
	ax.plot( output[:,2], range(0, 200), 'x', color='m', label='n=|3|')
	ax.plot( output[:,3], range(0, 200), 'x', color='c', label='n=|4|')
	ax.legend()




	scale = 0.06632
	ticks = np.array([0,\
					arr.shape[1]*0.25,\
					arr.shape[1]*0.5,\
					arr.shape[1]*0.75,\
					arr.shape[1]])

	labels = np.array([ str(0),\
	        str(np.round(arr.shape[1]*0.25*scale,2)),\
	        str(np.round(arr.shape[1]*0.5*scale,2)),\
	        str(np.round(arr.shape[1]*0.75*scale,2)),\
	        str(np.round(arr.shape[1]*scale,2))])

	ax.set_xticks( ticks, labels=labels)

	scale = 6.25
	ticks = np.array([0,\
					arr.shape[0]*0.25,\
					arr.shape[0]*0.5,\
					arr.shape[0]*0.75,\
					arr.shape[0]])

	labels = np.array([str(np.round(-arr.shape[0]*0.5*scale,2)),\
	        str(np.round(-arr.shape[0]*0.25*scale,2)),\
	        str(0),
	        str(np.round(arr.shape[0]*0.25*scale,2)),\
	        str(np.round(arr.shape[0]*0.5*scale,2))])

	ax.set_yticks( ticks, labels=labels)

	# then fit parabola to get equation for measuring defocus.

	def hyperbola( x, a, b, c, d ):
		y = (a/(b*x)) + c
		return y

	k = 100
	temp=output[0:k,:]

	ydata = range(-100, 0)

	from scipy.optimize import curve_fit
	popt, pcov = curve_fit( hyperbola,
							temp[:,0],
							ydata,
							maxfev=5000,
							nan_policy='omit' )

	fig, ax = plt.subplots()
	ax.plot( output[:,0], range(-100, 100),  'x', color='k', label='n=|1|')
	ax.plot(hyperbola(range(0, k), *popt), 'g--', label='best fit' )
	#ax.plot(range(0, k), (hyperbola(range(0, k), *popt)), 'g--', label='best fit' )
	#ax.set_xlim([0, 40])
	#ax.set_ylim([ydata[0], ydata[-1]])
	
	
	#temp=output[k:200,:]
	#.ydata = range(0, 100)

	popt, pcov = curve_fit( hyperbola,
							temp[:,0],
							ydata,
							maxfev=5000,
							nan_policy='omit' )
	ax.plot(ydata, hyperbola(range(k, 200), *popt), 'r--', label='best fit' )

	

	return

# Script starts here.
print('Starting tests.')

path = 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\'

#_test_astig_measure( path+'example_astigmatism2.tif' )

#_TFS( path+'\\wip\\example_TFS.tif' )
_test_zer( path+'\\wip\\TFS.npy' )



#_test_1D_sim()
#_test_2D_sim()
#_test_show_image( path+'example_image.tif' )
#_test_defocus( path + 'example_CTF.tif' )
#_test_profiles( path + 'example_CTF.tif' )
#_test_background_subtraction( path+'example_CTF.tif' )
#_test_astig( 'assets\\example_astigmatism.tif' )
#_test_chromatic()
#_test_other( path+'example_CTF.tif' )
#_test_fourier('assets\\example_image.tif')
#_test_TFS( 'assets\\example_TFS.tif' )
#_test_custom( 'C:\\Users\\pczbw2\\Desktop\\git\\pyCTF\\assets\\test_img.tif' )
'''
from pyCTF.figures import zernike_polynomials

zernike = zernike_polynomials()

size = 1024

arr = np.ones([size, size])

from pyCTF.utils import find_iradius_itheta

iradius, itheta = find_iradius_itheta( arr, 1 )

from pyCTF.image import ElectronImage
from pyCTF.utils import show_image
from pyCTF.utils import normalise_data_range

	
CTF = pyCTF.image.import_ctf( np.array( Image.open( path + 'example_CTF.tif'  )), 
												200, 
												0.0066127 )
CTF.C12 = 50e-9
CTF.defocus = 1000e-9
CTF.Cs = 1.0e-3

#polynomials = zernike._phase_plate( CTF, CTF.image.shape[0]/2, CTF.iradius, CTF.itheta )
polynomials = zernike._phase_plate( arr, arr.shape[0]/2, iradius, itheta )

#fig, ax = zernike._plot_phase_plate( polynomials, 0.01 )
fig, ax = zernike._plot_all(polynomials)
'''
print('Tests finished.')

plt.show()