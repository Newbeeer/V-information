#Copyright Weihao Gao, UIUC

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import numpy as np
import torch
from model import mlp,mlp_try
from data import *
#Mixed_KSG Algorithm
def Mixed_KSG(x,y,k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using *Mixed-KSG* mutual information estimator

		Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
		y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
		k: k-nearest neighbor parameter

		Output: one number of I(X;Y)
	'''

	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		kp, nx, ny = k, k, k
		if knn_dis[i] == 0:
			kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
			nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
		else:
			nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
		ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
	return ans


#Partitioning Algorithm
def Partitioning(x,y,numb=8):
	#assert len(x)==len(y), "Lists should have same length"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])

	minx = np.zeros(dx)
	miny = np.zeros(dy)
	maxx = np.zeros(dx)
	maxy = np.zeros(dy)
	for d in range(dx):
		minx[d], maxx[d] = x[:,d].min()-1e-15, x[:,d].max()+1e-15
	for d in range(dy):
		miny[d], maxy[d] = y[:,d].min()-1e-15, y[:,d].max()+1e-15

	freq = np.zeros((numb**dx+1,numb**dy+1))
	for i in range(N):
		index_x = 0
		for d in range(dx):
			index_x *= dx
			index_x += int((x[i][d]-minx[d])*numb/(maxx[d]-minx[d]))
		index_y = 0
		for d in range(dy):
			index_y *= dy
			index_y += int((y[i][d]-miny[d])*numb/(maxy[d]-miny[d]))
		freq[index_x][index_y] += 1.0/N
	freqx = [sum(t) for t in freq]
	freqy = [sum(t) for t in freq.transpose()]
	
	ans = 0
	for i in range(numb**dx):
		for j in range(numb**dy):
			if freq[i][j] > 0:
				ans += freq[i][j]*log(freq[i][j]/(freqx[i]*freqy[j]))
	return ans

#Noisy KSG Algorithm
def Noisy_KSG(x,y,k=5,noise=0.01):
	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)
	
	if noise > 0:
		data += nr.normal(0,noise,(N,dx+dy))

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
		ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
		ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
	return ans

#Original KSG estimator
def KSG(x,y,k=5):
	#assert len(x)==len(y), "Lists should have same length"
	#assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)
	
	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans = 0

	for i in range(N):
		nx = len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=float('inf')))-1
		ny = len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=float('inf')))-1
		ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny))/N
	return ans

def F_linear_gaussian(x,y):

	#F_informaiton from x to y (I_F(x \to y)

	N = len(x)

	if x.ndim == 1:
		x = x.reshape((N, 1))
	dx = len(x[0])
	if y.ndim == 1:
		y = y.reshape((N, 1))
	dy = len(y[0])
	ones = np.ones((N,1))
	x = np.concatenate(( x**3, x**2, x, ones), axis=1)
	H_y = np.var(y)

	t = x.dot(np.linalg.inv(x.T.dot(x))).dot(x.T).dot(y)

	H_y_x = ((y - t) ** 2).sum()/N

	return H_y- H_y_x



def F_discrete_categorical(x,y,N):


	'''
	:param x:
	:param y:
	:param N: size of alphabet
	:return:
	'''

	len = y.shape[0]

	bins = np.zeros((N))
	for i in range(N):
		bins[i] = float((y==i).sum())

	p = 0.0
	for i in range(N):
		if bins[i] == 0:
			continue
		p += np.log(bins[i]/len) * (bins[i]/len)

	H_y = -1 * p

	bins = np.zeros((N,N)) #row:x col:y
	xbins = np.zeros((N))
	for x_i in range(N):
		xbins[x_i] = (x==x_i).sum()
		for y_i in range(N):

			bins[x_i][y_i] = ((x==x_i) * (y==y_i)).sum()

	#bins =  bins / bins.sum(1)[:,None]
	xbins = xbins / len
	H_y_x = 0.0
	for x_i in range(N):
		p = 0.0
		normalize = bins[x_i].sum()
		if normalize == 0:
			continue
		for y_i in range(N):
			if bins[x_i][y_i] == 0:
				continue
			p += np.log(bins[x_i][y_i]/normalize) * (bins[x_i][y_i]/normalize)
		H_y_x += xbins[x_i] * (p)

	H_y_x = -1 * H_y_x

	return H_y - H_y_x


def F_mlp_gaussian(x, y):
	# F_informaiton from x to y (I_F(x \to y)

	model = mlp_try(1).cuda()
	opt = torch.optim.SGD(model.parameters(), lr=0.0001)
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N, 1))
	if y.ndim == 1:
		y = y.reshape((N, 1))
	x = np.concatenate([x**3,x**2,x],axis=1)
	dataset = Two_Random(x,y)
	dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 256, shuffle = True)
	mu = np.sum(y, axis=0) / N
	H_y = np.sum((y - mu) ** 2) / N

	epoch = 80
	#print("-------------------")
	model.train()
	for i in range(epoch):
		train_loss = 0.0
		for x_,y_ in dataloader:
			x_ = x_.cuda().float()
			y_ = y_.cuda().float()
			opt.zero_grad()
			print(x_.size())
			output = model(x_)
			loss = torch.pow(output - y_, 2).mean()
			print("Loss:",loss.item())
			train_loss += loss.item()

			loss.backward()
			opt.step()
		print("-----Loss:", train_loss)

	H_y_x = 0.0
	model.eval()
	for x_, y_ in dataloader:
		x_ = x_.cuda().float()
		y_ = y_.cuda().float()
		output = model(x_)
		loss = torch.pow(output - y_, 2).sum()
		H_y_x += loss.item()

	H_y_x = H_y_x / N

	print( H_y - H_y_x)
	return H_y - H_y_x







