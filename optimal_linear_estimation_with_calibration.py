#!/usr/bin/env python

###############################################################
### OPTIMAL LINEAR ESTIMATION with CALIBRATION ###
#
# Copyleft 2019 Vasiliy Pushkarev (https://github.com/vvasill)
#
# This work may be distributed and/or modified under the
# conditions of the GNU General Public License v3.0
###############################################################

import os
import random
import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt
import matplotlib.cm as cm

####################################################################
### PARAMETERS ###
### list of observation numbers
n_obs_list = np.array([1, 10, 100, 1000])
#n_obs_list = np.arange(1, 21, 1)
### list of calibrating signals numbers
n_calib_list = np.array([100, 1000, 10000])
#n_calib_list = np.arange(100, 2100, 100)
### x-interval
x_left = -10.0
x_right = 10.0
### mu params
mu_av = 0.0
mu_sigma = 1.0
### nu params
nu_av = 0.0
nu_sigma = 10.0
### gaussian params
ampl = 0.5	
mean = 0.0
stddev = 0.7
### A-matrix params
Aij_min = -5 #min value in A-matrix
Aij_max = 5 #max value in A-matrix
ysize = 100 #vertical size of A-matrix
xsize = 50 #horizontal size of A-matrix
### randomize
random.seed(version=2)

####################################################################
### FUNCTIONS ###
def gaussian(x_dots, amplitude, mean, stddev):
    return amplitude * np.exp(-((x_dots - mean) / 4.0 / stddev)**2)

def random_number(av, sigma):
	### generate identically distributed numbers with preset average and sigma
	number = 2*sigma*np.random.random() - sigma + av
	### or generate normally distributed numbers with preset average and sigma
	#number = np.random.normal(av, sigma)
	return number

def gen_smooth_random_signal(x_dots, mu_av, mu_sigma, ampl, mean, stddev):
	for i in range(xsize):
		mu[i] = random_number(mu_av, mu_sigma)
		b[i] = gaussian(x_dots[i], ampl, mean, stddev)
	### Toeplitz matrix generation
	B = lng.toeplitz(b)
	### smoothing
	x = np.matmul(B, mu)
	F = np.matmul(B, np.transpose(B))
	return (x, F)

def calibtocan(psi, fi):
	G = np.outer( psi, np.transpose(fi) )
	H = np.outer( fi, np.transpose(fi) )
	return G, H

def calib_BLUE(G, H, u, S, n, x0, F):
	A0 = np.matmul(G, np.linalg.inv(H))
	alpha = np.trace(np.matmul(np.linalg.inv(H), F))
	J = alpha*S
	Q = np.linalg.inv(np.matmul(np.matmul(np.transpose(A0), np.linalg.inv(J + S/n)), A0) + np.linalg.inv(F))
	R = np.matmul(np.matmul(Q, np.transpose(A0)), np.linalg.inv(J + S/n))
	r = np.matmul(Q, np.matmul(np.linalg.inv(F), x0))
	x_est = np.matmul(R, u/n) + r
	var_est = np.diagonal(Q)
	return x_est, var_est

def plot(x_dots, x, x_est, var_est, outfile):
	plt.rcParams["figure.figsize"] = (10,7)
	plt.rc('font', size=18) 
	plt.plot(x_dots, x, 'g-', label='x')
	plt.plot(x_dots, x_est, 'r-', label='x_est')
	plt.plot(x_dots, x_est - np.sqrt(var_est), 'r--', label='stddev')
	plt.plot(x_dots, x_est + np.sqrt(var_est), 'r--')
	plt.legend(loc='best', prop={'size': 18})
	plt.grid(True)
	plt.savefig(outfile)
	plt.close()

def plot_contours(x_dots, y_dots, f, outfile):
	cs = plt.contour(x_dots, y_dots, f, 35)
	#plt.clabel(cs, inline=1, fontsize=10)	
	plt.colorbar(cs, shrink=0.8, drawedges='0', extendfrac='1, 1')	
	plt.rcParams["figure.figsize"] = (10,8)
	plt.rc('font', size=18) 
	plt.xlabel(r'$n_{obs}$')
	plt.ylabel(r'$k$')
	plt.grid(True)
	plt.savefig(outfile)
	plt.close()

def plot_heatmap(data, outfile):
	plt.imshow(data, cmap='magma', interpolation='nearest')
	plt.colorbar()	
	plt.rc('font', size=18) 
	plt.xlabel(r'$n_{obs}$')
	plt.ylabel(r'$k$')
	plt.grid(True)
	plt.savefig(outfile)
	plt.close()

####################################################################
### some initial assignments
x_dots = np.arange(x_left, x_right, (x_right-x_left)/xsize)
mu = np.zeros([xsize])
nu = np.zeros([ysize])
b = np.zeros([xsize])
A = np.zeros([ysize, xsize])
T = np.zeros([xsize, xsize])
v = np.zeros([xsize])
### apriori information generation
x0 = np.zeros([xsize])
### directory for images
if not os.path.exists('./images'):
    os.makedirs('./images')

####################################################################
### SMOOTH RANDOM SIGNAL GENERATION ###
(x, F) = gen_smooth_random_signal(x_dots, mu_av, mu_sigma, ampl, mean, stddev)

### signal graph plotting
plt.rcParams["figure.figsize"] = (10,7)
plt.rc('font', size=18) 
plt.plot(x_dots, mu, 'r.', label='mu')
plt.plot(x_dots, b, 'b-', label='gaussian')
plt.plot(x_dots, x, 'g-', label='smooth signal')
plt.legend(loc=0, prop={'size': 18})
plt.grid(True)
plt.savefig('./images/x_signal.eps')
plt.close()

####################################################################
### OPTIMAL LINEAR ESTIMATION for SEVERAL OBSERVATIONS with CALIBRATION ###
traceQ = np.zeros([n_calib_list.shape[0], n_obs_list.shape[0]])

for kk in range(n_calib_list.shape[0]):
	n_calib = n_calib_list[kk]
	for nn in range(n_obs_list.shape[0]):
		n_obs = n_obs_list[nn]
		print('n_obs = ' + str(n_obs) + ', n_calib = ' + str(n_calib))

		####################################################################
		### OBSERVATIONS INFORMATION GENERATION ###
		u = np.zeros([ysize])

		### raw information generation: generate A-matrix
		for i in range(ysize):
			for j in range(xsize):
				A[i][j] = np.random.randint(Aij_min, Aij_max)

		for j in range(n_obs):
			### raw information generation: generate new observation
			for i in range(ysize):
				nu[i] = random_number(nu_av, nu_sigma)
			y = np.matmul(A, x) + nu
			S = nu_sigma**2*np.identity(ysize)
			### canonical information updating
			u += y

		####################################################################
		### CALIBRATING INFORMATION GENERATION ###
		G = np.zeros([ysize, xsize])
		H = np.zeros([xsize, xsize])

		for j in range(n_calib):
			### calibrating signal generating ###
			(fi, calib_F) = gen_smooth_random_signal(x_dots, mu_av, mu_sigma, ampl, mean, stddev)

			### raw information generation: generate new calibration
			for i in range(ysize):
				nu[i] = random_number(nu_av, nu_sigma)
			psi = np.matmul(A, fi) + nu
			fi_S = nu_sigma**2*np.identity(ysize)
			### raw to canonical: elementary information calculation
			(Gi, Hi) = calibtocan(psi, fi)
			### canonical information updating
			G += Gi
			H += Hi

		####################################################################
		### OPTIMAL LINEAR ESTIMATION ###
		### BLUE (Best Linear Unbiased Estimation) approximation with calibration ###
		### (x0, F) --- priori info
		### (u, S, n_obs) --- accumulated observational info
		### (G, H) --- calibrating canonial info
		(x_est, var_est) = calib_BLUE(G, H, u, S, n_obs, x0, F)
		### graph plotting
		fig_name = './images/n=' + str(n_obs) + 'k=' + str(n_calib) + '.eps'
		plot(x_dots, x, x_est, var_est, fig_name)
		fig_name = './images/n=' + str(n_obs) + 'k=' + str(n_calib) + '.png'
		plot(x_dots, x, x_est, var_est, fig_name)
			
		####################################################################
		### trace(Q) --- full estimation error --- calculation ###
		for i in range(var_est.shape[0]):
			traceQ[kk][nn] += var_est[i]	

####################################################################
### trace(Q) vizualization ###
fig_name = './images/traceQ.eps'
#plot_contours(n_obs_list, n_calib_list, traceQ, fig_name)
fig_name = './images/traceQ.png'
#plot_contours(n_obs_list, n_calib_list, traceQ, fig_name)
