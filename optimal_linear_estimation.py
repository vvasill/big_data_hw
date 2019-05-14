#!/usr/bin/env python

###############################################################
### OPTIMAL LINEAR ESTIMATION ###
#
# Copyleft 2019 Vasiliy Pushkarev (https://github.com/vvasill)
#
# This work may be distributed and/or modified under the
# conditions of the GNU General Public License v3.0
###############################################################

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng

####################################################################
### INFORMATION ###
# (y, A, S) --- raw format 
# (x, F) --- explicit format
# (T, v) --- canonical format

####################################################################
### FUNCTIONS ###
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def rawtocan(y, A, S):
	T = np.matmul( np.matmul( np.transpose(A), np.linalg.inv(S) ), A )
	v = np.matmul( np.matmul( np.transpose(A), np.linalg.inv(S) ), y )
	return T, v

def BLUE(y, A, S):
	D = np.linalg.inv(np.matmul(np.matmul(np.transpose(A), np.linalg.inv(S)), A))
	x_est = np.matmul( np.matmul( np.matmul( D, np.transpose(A) ), np.linalg.inv(S) ), y )
	var_est = np.diagonal(D)
	return x_est, var_est

def LSTQ(T, v):
	D = np.linalg.inv(T)
	x_est = np.matmul(D, v)
	var_est = np.diagonal(D)
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

####################################################################
### PARAMETERS ###
### x-interval
x_left = -10.0
x_right = 10.0
### list of observation numbers
n_obs_list = np.array([1, 10, 100])
### mu params
mu_av = 0.0
mu_sigma = 1.0
### nu params
nu_av = 0.0
nu_sigma = 10.0
### gaussian params
ampl = 0.5	
mean = 0.0
stddev = 1.0
### A-matrix params
Aij_min = -5 #min value in A-matrix
Aij_max = 5 #max value in A-matrix
ysize = 100 #vertical size of A-matrix
xsize = 50 #horizontal size of A-matrix
### randomize
random.seed(version=2)

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

####################################################################
### SMOOTH RANDOM SIGNAL GENERATION ###
for i in range(xsize):
	mu[i] = np.random.normal(mu_av, mu_sigma)
	b[i] = gaussian(x_dots[i], ampl, mean, stddev)
### Toeplitz matrix generation
B = lng.toeplitz(b)
### smoothing
x = np.matmul(B, mu)
F = np.matmul(B, np.transpose(B))

### graph plotting
plt.rcParams["figure.figsize"] = (10,7)
plt.rc('font', size=18) 
plt.plot(x_dots, mu, 'r.', label='mu')
plt.plot(x_dots, b, 'b-', label='gaussian')
plt.plot(x_dots, x, 'g-', label='smooth signal')
plt.legend(loc=0, prop={'size': 18})
plt.grid(True)
plt.savefig('x_signal.png')
plt.close()

####################################################################
### SEVERAL OBSERVATIONS ###

for n in range(n_obs_list.shape[0]):
	n_obs = n_obs_list[n]
	print(n_obs)
	T = np.zeros([xsize, xsize])
	v = np.zeros([xsize])

	for k in range(n_obs):
		### raw information generation: generate new observation
		for i in range(ysize):
			nu[i] = np.random.normal(nu_av, nu_sigma)
			for j in range(xsize):
				A[i][j] = np.random.randint(Aij_min, Aij_max)
		y = np.matmul(A, x) + nu
		S = nu_sigma**2*np.identity(ysize)
		### raw to canonical: elementary information calculation
		(Ti, vi) = rawtocan(y, A, S)
		### canonical information updating
		T += Ti
		v += vi

	### BLUE (Best Linear Unbiased Estimation) approximation ###
	#(x_est, var_est) = BLUE(y, A, S)
	### LSTQ (least squares) approximation ###
	(x_est, var_est) = LSTQ(T, v)
	### graph plotting
	fig_name = 'n=' + str(n_obs)
	plot(x_dots, x, x_est, var_est, fig_name)

	####################################################################
	### SEVERAL OBSERVATIONS with APRIORI INFORMATION ###
	### apriori information to canonical information
	(T0, v0) = (np.linalg.inv(F), np.matmul(np.linalg.inv(F), x0))
	### joining apriori information in canonical form and elementary information
	T += T0
	v += v0
	### LSTQ (least squares) approximation ###
	(x_est, var_est) = LSTQ(T, v)
	### graph plotting
	fig_name = 'n=' + str(n_obs) + '_with_aprior.png'
	plot(x_dots, x, x_est, var_est, fig_name)
