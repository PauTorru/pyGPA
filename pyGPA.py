import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
import skimage.filters
import skimage.util
import scipy.optimize
import hyperspy.api as hs
import scipy.optimize



def select_spots(im):
	r"""Select spots for gpa.
	Parmaters:
	----------
	im : hs.signals.Signal2D
	High resolution image suitable for GPA.
	"""

	fft = np.fft.fftshift(np.fft.fft2(im.data))
	psd = np.abs(fft)**2

	im_fft = hs.signals.Signal2D(np.log(psd))
	im_fft.plot()
	sf,sc =im_fft.data.shape
	c1 = hs.roi.CircleROI(sc//4,sf//2,sc//6)
	c2 = hs.roi.CircleROI(sc//2,sf//4,sc//6)
	
	c1.interactive(im_fft,color="red")
	c2.interactive(im_fft,color="red")

	return c1,c2



def select_reference_roi(im):
	r"""Select reference region for gpa.
	Parmaters:
	----------
	im : hs.signals.Signal2D
	High resolution image suitable for GPA.
	"""
	rf = hs.roi.RectangularROI()
	im_nocal = hs.signals.Signal2D(im.data)
	im_nocal.plot()
	rf.interactive(im_nocal,color="red")
	return rf


def GPA(image,positions,reference,sigma = 2):
	r"""retruns e_xx e_yy shear and rotation"""

	if all([isinstance(p,hs.roi.CircleROI) for p in positions]):
		r = np.average([c.r for c in positions])
		ps = [(c.cx,c.cy,r) for c in positions]
	else:
		ps = positions

	if isinstance(reference,hs.roi.RectangularROI):
		fi = int(reference.top)
		ff = int(reference.bottom)
		ci = int(reference.left)
		cf = int(reference.right)
	else:
		fi,ff,ci,cf = reference



	fft = np.fft.fftshift(np.fft.fft2(image))
	psd = np.abs(fft)**2

	gmatrix = []
	phagrads = []


	for i,p in enumerate(ps):
		x,y,r = p
		mask = np.zeros_like(psd)
		rr, cc = skimage.draw.disk([y,x],r,shape=mask.shape)
		mask[rr, cc] = 1
		mask = skimage.filters.gaussian(mask,sigma)

		invfft = np.fft.ifft2(np.fft.fftshift(fft*mask))
		invfftamp = np.abs(invfft)
		invfftpha = np.angle(invfft)

		phacut = invfftpha[fi:ff,ci:cf]
		xs = np.arange(ci,cf)
		ys = np.arange(fi,ff)
		xx,yy = np.meshgrid(xs,ys)


		fun = lambda x: np.sum(np.unwrap(phacut - 2*np.pi*(x[0]*xx+x[1]*yy-x[2]),axis=i-1)**2)
		opt = scipy.optimize.minimize(fun,[0,0,0],method = 'Nelder-Mead')

		gx = opt.x[0]
		gy = opt.x[1]
		gc = opt.x[2]

		xs = np.arange(invfftpha.shape[1])
		ys = np.arange(invfftpha.shape[0])
		xx,yy = np.meshgrid(xs,ys)
		reduced_phase = np.mod(invfftpha - 2*np.pi*(gx*xx+gy*yy-gc)-np.pi,2*np.pi)
		epha = np.exp(1j*reduced_phase)
		ephainv = np.exp(-1j*reduced_phase)
		phagradx = np.imag(ephainv*np.gradient(epha))[1,:,:]
		phagrady = np.imag(ephainv*np.gradient(epha))[0,:,:]


		phagrads.append([phagradx,phagrady])
		gmatrix.append([gx,gy])



	gmatrix = np.array(gmatrix)
	amatrix = np.linalg.inv(gmatrix)
	phagradmatrix = np.array(phagrads)


	ematrix = -(1/(2*np.pi))*np.tensordot(amatrix,phagradmatrix,1)
	epsmatrix = 0.5*(ematrix+np.swapaxes(ematrix,0,1))

	In_plain_strain = epsmatrix[0,0,:,:]
	Out_plain_strain = epsmatrix[1,1,:,:]
	Shear = 0.5*(ematrix[1,0,:,:]+ematrix[0,1,:,:])
	Rotation = np.rad2deg(np.arcsin(0.5*(ematrix[1,0,:,:]-ematrix[0,1,:,:])))

	return (In_plain_strain, Out_plain_strain, Shear, Rotation)


def chop_outliers(im,bins = 100,remove =1):
	if remove ==0:
		return im.copy()
	if isinstance(remove,list):
		mr = remove[0]
		Mr = remove[1]
	else:
		mr = remove
		Mr = -remove

	density, weights = np.histogram(im.ravel(),bins=bins)
	out = im.copy()
	out[out<weights[mr]]=weights[mr]
	out[out>weights[Mr]]=weights[Mr]
	return out
















