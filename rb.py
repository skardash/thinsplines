import numpy
import math
import numpy.matlib
import scipy.io
import statistics
from numpy.linalg import inv

def radialBasis(ri):
	rli = ri
	sz = rli.shape
	ko = numpy.zeros(sz)
	for i in range(0,sz[0]):
		for j in range(0,sz[1]):
			if (rli[i,j] == 0):
				rli[i,j] = 2.2251e-308
			ko[i,j] = 2*(rli[i,j]*rli[i,j])*math.log(rli[i,j])
	return ko

def computeWl(xp, yp, np):
	rXp = numpy.tile(xp.transpose(),[1,np])
	rYp = numpy.tile(yp.transpose(),[1,np])
	sz = rXp.shape
	wR = numpy.asmatrix(numpy.zeros(sz))
	for i in range(0,sz[0]):
		for j in range(0,sz[1]):
			rXpt = rXp.transpose()
			rYpt = rYp.transpose()
			wR[i,j] = math.sqrt(pow(rXp[i,j]-rXpt[i,j],2)+pow(rYp[i,j]-rYpt[i,j],2)) # compute r(i,j)

	wK = radialBasis(wR)
	
	p1 = numpy.ones([np,1])
	p2 = xp.flatten('F').reshape(xp.size,1)
	p3 = yp.flatten('F').reshape(yp.size,1)
	wP = numpy.concatenate((p1, p2, p3), axis=1)

	p1 = numpy.concatenate((wK, wP), axis=1)
	p2 = numpy.concatenate((wP.transpose(), numpy.zeros([3,3])), axis=1)
	wL = numpy.concatenate((p1,p2), axis=0)

	return wL

def tpsMap(wW, imgH, imgW, xp, yp, np):
	x = numpy.linspace(1, imgH, imgH)
	y = numpy.linspace(1, imgW, imgW)
	X, Y = numpy.meshgrid(x, y)
	X = X.flatten('F')
	Y = Y.flatten('F')

	NWs = X.size
	rX = numpy.tile(X,[np,1]) # Np x NWs
	rY = numpy.tile(Y,[np,1]) # Np x NWs

	rxp = numpy.tile(xp.reshape(xp.size,1), [1,NWs]) #  1xNp to Np x NWs
	ryp = numpy.tile(yp.reshape(yp.size,1), [1,NWs]) # 1xNp to Np x NWs

	sz = rxp.shape
	wR = numpy.zeros(sz)
	for i in range(0,sz[0]):
		for j in range(0,sz[1]):
			wR[i,j] = math.sqrt(pow(rxp[i,j]-rX[i,j],2) + pow(ryp[i,j]-rY[i,j],2)) # distance measure r(i,j)=|Pi-(x,y)|

	wK = radialBasis(wR) # compute [K] with elements U(r)=r^2 * log (r^2)

	p1 = numpy.asmatrix(numpy.ones(NWs)).transpose()
	p2 = X.flatten('F')
	p3 = Y.flatten('F')
	wP = numpy.concatenate((p1, p2.reshape(p2.size,1), p3.reshape(p3.size,1)), axis=1).transpose() # % [P] = [1 x' y'] where (x',y') are n landmark points (nx2)
	wL = numpy.concatenate((wK, wP), axis=0).transpose() # [L] = [[K P];[P' 0]]
	
	wW1 = wW[:,0]
	wW1 = wW1.reshape(wW1.size,1)
	wW2 = wW[:,1]
	wW2 = wW2.reshape(wW2.size,1)

	Xw  = wL*wW1 #[Pw] = [L]*[W]
	Yw  = wL*wW2 # [Pw] = [L]*[W]
	return (Xw, Yw)

def tpswarp(img, outDim, Zp, Zs, interp):
	NPs = Zp.shape[0]
	imgH = img.shape[0]
	imgW = img.shape[1]

	outH = outDim[0, 1]
	outW = outDim[0, 0]

	Xp = Zp[:,0].reshape(1, Zp.shape[0])
	Yp = Zp[:,1].reshape(1, Zp.shape[0])

	Xs = Zs[:,0].reshape(1, Zs.shape[0])
	Ys = Zs[:,1].reshape(1, Zs.shape[0])

	wL=computeWl(Xp, Yp, NPs)

	p1 = Xs.flatten('F').reshape(Xs.size,1)
	p2 = Ys.flatten('F').reshape(Ys.size,1)
	p3 = numpy.zeros([3,2])
	p12 = numpy.concatenate((p1, p2), axis=1)
	wY = numpy.concatenate((p12, p3), axis=0)

	wW = numpy.dot(inv(wL),wY)

	(Xw, Yw)=tpsMap(wW, imgH, imgW, Xp, Yp, NPs)
	x = numpy.linspace(1, imgH, imgH)
	y = numpy.linspace(1, imgW, imgW)
	X, Y = numpy.meshgrid(x, y)

	Xm = X.flatten('F').reshape(X.size,1)
	Ym = Y.flatten('F').reshape(Y.size,1)
	(imgw,imgwr,mp) = interp2d(Xm, Ym, img, Xw, Yw, outH, outW, interp)
	return (imgw,imgwr,mp)

def getd(a,ix):
	at = numpy.asarray(a.flatten('F')).transpose()
	ixt = ix.transpose().tolist()[0]
	return numpy.asmatrix(at[ixt]).transpose()

def sub2ind(sizes, index_x, index_y):
    res = numpy.zeros(index_x.shape)
    for i in range(0, index_x.shape[0]):
        res[i,0] = sizes[0]*index_y[i,0] + index_x[i,0]
    return res

def assign(a,ix,b):
	at = a.flatten('F')
	ixt = ix.transpose().tolist()
	at[ixt] = b 
	res = numpy.reshape(at, a.shape, order='F')
	return res

def min_spec(vec,scal):
	#res = numpy.asmatrix(numpy.zeros(vec.shape))
	res = numpy.zeros(vec.shape)
	for i in range(0, vec.shape[0]):
		res[i,0]=min(vec[i,0],scal)
	return res

def max_spec(vec,scal):
	#res = numpy.asmatrix(numpy.zeros(vec.shape))
	res = numpy.zeros(vec.shape)
	for i in range(0, vec.shape[0]):
		res[i,0]=max(vec[i,0],scal)
	return res

def interp2d(X, Y, img, Xwr, Ywr, outH, outW, interp):
	color = img.shape[2]
	imgwr = numpy.zeros([outH,outW,color])
	# window dimension for filling

	maxhw = (interp['radius'][0,0][0,0]-1)/2

	color = img.shape[2]
	imgH  = img.shape[0]
	imgW  = img.shape[1]

	Xwr = Xwr - 1
	Ywr = Ywr - 1

	X = X-1
	Y = Y-1

	Xwi = Xwr.round()
	Ywi = Ywr.round()

	# Bound warped coordinates to image frame
	Xwi = max_spec(min_spec(Xwi,outH-1),0);
	Ywi = max_spec(min_spec(Ywi,outW-1),0);

	# Convert 2D coordinates into 1D indices
	fiw = sub2ind([outH,outW],Xwi,Ywi).astype(int) # warped coordinates
	fip = sub2ind([imgH,imgW],X,Y).astype(int) # input

	scipy.io.savemat('fiw.mat', {'fiw':fiw})

	o_r = numpy.zeros([outH,outW])
	for colIx in range(0,color):
		img_r = img[:,:,colIx]
		o_r = assign(o_r,fiw,getd(img_r,fip))
		imgwr[:,:,colIx]=o_r

	mp = numpy.zeros([outH,outW])
	mp = assign(mp,fiw,1)

	if interp['method'] == 'nearest':
	    imgw = nearestInterp(imgwr, mp, maxhw);
	elif interp['method'] =='invdist':
	    imgw = idwMvInterp(imgwr, mp, maxhw, interp['power'])
	else:
		imgw = imgwr

	return (imgw,imgwr,mp)

# nearestInterp

def nearestInterp(imgw, mp, maxhw):
	outH  = imgw.shape[0] # size(imgw,1);
	outW  = imgw.shape[1] #size(imgw,2);
	out = imgw.astype(float)
	yi_arr, xi_arr = numpy.where(mp==0)
	yi_arr = yi_arr.reshape(yi_arr.shape[0],1)
	xi_arr = xi_arr.reshape(xi_arr.shape[0],1)
	if yi_arr.shape[0] > 0:
		color = imgw.shape[2]
		for ix in range(0,yi_arr.shape[0]):
			xi = xi_arr[ix,0]
			yi = yi_arr[ix,0]
			nz = False
			for h in range(1,maxhw+1):
				yixL = max(yi-h,0)
				yixU = min(yi+h,outH-1)
				xixL = max(xi-h,0)
				xixU = min(xi+h,outW-1)
				mapr = mp[yixL:yixU+1,xixL:xixU+1]
				i,j = numpy.where(mapr==1)
				if i.shape[0] > 0:
					nz = True;
					break

			if nz == True:
				for colIx in range(0,color):
					win=imgw[yixL:yixU+1, xixL:xixU+1, colIx]
					mapr = mp[yixL:yixU+1, xixL:xixU+1]
					i,j = numpy.where(mapr!=0)
					lst = []
					for k in range(0,i.shape[0]):
						lst.append(win[i[k],j[k]])
					out[yi,xi,colIx] = statistics.median(lst)
	return out

# idwMvInterp
def compWk(mp, cx, cy, p):
	h = mp.shape[0]
	w = mp.shape[1]
	xx = numpy.linspace(1, h, h)
	yy = numpy.linspace(1, w, w)
	x, y = numpy.meshgrid(xx, yy)
	y = numpy.power(y-cx, 2)
	x = numpy.power(x-cy, 2)
	d2 = x + y
	wk = numpy.divide(1,numpy.power(d2.transpose(), p/2)) 
	return wk

def find(x):
	i,j = numpy.where(x != 0)
	ix = numpy.sort(i + j*x.shape[0]).reshape(i.shape[0],1)
	return ix

def findn(x,n):
	ix = find(x)
	if n>ix.shape[0]:
		n = ix.shape[0]
	return ix[0:n]

def isempty(x):
	return x.size == 0

def idw(in_c, mp, wk):
	mpf = find(mp)
	p1 = getd(in_c,mpf)
	p2 = getd(wk,mpf)
	num = numpy.sum(numpy.multiply(p1,p2))
	den=numpy.sum(getd(wk,mpf))	
	out = num / den
	return out

def idwMvInterp(imgw, mp, maxhw, p):
	outH  = imgw.shape[0] # size(imgw,1);
	outW  = imgw.shape[1] #size(imgw,2);
	out = imgw.astype(float)
	yi_arr, xi_arr = numpy.where(mp==0)
	if yi_arr.shape[0] > 0:
		color = imgw.shape[2]
		for ix in range(0,yi_arr.shape[0]):
			xi = xi_arr[ix]
			yi = yi_arr[ix]

			yixL = max(yi-maxhw,0)
			yixU = min(yi+maxhw,outH-1)
			xixL = max(xi-maxhw,0)
			xixU = min(xi+maxhw,outW-1)

			mapw = mp[yixL:yixU+1, xixL:xixU+1]
			if not(isempty(findn(mapw,1))):
				wk = compWk(mapw, xi-xixL+1, yi-yixL+1, p)
				for colIx in range(0,color):
					out[yi,xi,colIx] = idw(imgw[yixL:yixU+1, xixL:xixU+1, colIx], mapw, wk)
	return out

if 1:
	mat = scipy.io.loadmat('tpswarp.mat')
	(imgw,imgwr,mp) = tpswarp(mat['img'], mat['outDim'], mat['Zp'], mat['Zs'], mat['interp'])
	numpy.savetxt('tpswarp_imgw_out0_py.txt', imgw[:,:,0], fmt='%.2f')	
	numpy.savetxt('tpswarp_imgwr_out0_py.txt', imgwr[:,:,0], fmt='%.2f')	
	numpy.savetxt('tpswarp_mp_py.txt', mp, fmt='%.1f')	
if 0:
	mat = scipy.io.loadmat('computeWl.mat')
	wL = computeWl(mat['xp'],mat['yp'],mat['np'][0,0])
	numpy.savetxt('computeWl_py.txt', wL, fmt='%.3f')	

if 0:
	mat = scipy.io.loadmat('tpsMap.mat')
	(Xw, Yw) = tpsMap(mat['wW'], mat['imgH'][0,0], mat['imgW'][0,0], mat['xp'], mat['yp'], mat['np'][0,0])
	numpy.savetxt('tpsMap_Xw_py.txt', Xw, fmt='%.3f')	
	numpy.savetxt('tpsMap_Yw_py.txt', Yw, fmt='%.3f')	

if 0:
	mat = scipy.io.loadmat('interp2d.mat')
	(imgw,imgwr,mp) = interp2d(mat['X'], mat['Y'], mat['img'], mat['Xwr'], mat['Ywr'], mat['outH'][0,0], mat['outW'][0,0], mat['interp'])
	numpy.savetxt('interp2d_imgw_out0_py.txt', imgw[:,:,0], fmt='%.1f')	
	numpy.savetxt('interp2d_imgwr_out0_py.txt', imgwr[:,:,0], fmt='%.1f')	
	numpy.savetxt('interp2d_mp_py.txt', mp, fmt='%.1f')	

if 0:
	mat = scipy.io.loadmat('nearestInterp.mat')
	out = nearestInterp(mat['imgw'], mat['map'], mat['maxhw'][0,0])
	numpy.savetxt('out1_py.txt', out[:,:,0], fmt='%.1f')	

if 0:
	# idwMvInterp
	mat = scipy.io.loadmat('idwMvInterp.mat')
	out = idwMvInterp(mat['imgw'], mat['map'], mat['maxhw'][0,0], mat['p'])
	numpy.savetxt('out_idw1.txt', out[:,:,0], fmt='%.2f')
