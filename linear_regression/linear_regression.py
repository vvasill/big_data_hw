#!/usr/bin/env python

###############################################################
### LINEAR REGRESSION ###
#
# Copyleft 2019 Vasily Pushkarev (https://github.com/vvasill)
#
#This work may be distributed and/or modified under the
#conditions of the GNU General Public License v3.0
###############################################################

import random
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.abc import z

###############################################################
### FUNCTIONS ###
### function f(x)
def f(a, x):
	result = 0.0
	for i in range(a.shape[0]):
		result += a[i]*x**i
	return result

### data point generation
def gen(a, av, sigma, x_left, x_right):
	x = (abs(x_right) + abs(x_left))*random.random() + x_left
	eps = np.random.normal(av, sigma)
	y = f(a,x) + eps
	return (x, y)

###############################################################
### PARAMETERS ###
### polynomial degree
m = 5
### vector F and a-coefficients assignments 
F = Matrix([[1, z, z**2, z**3, z**4]])
a = np.array([1.0, 1.0, -2.0, 0.04, 0.02])
### number of x-points
nn = 200
### eps params
av = 0.0
sigma = 50.0
### x-interval
x_left = -10
x_right = 10
step = 0.01
### randomize
random.seed(version=2)

###############################################################
### initial assignments
T = Matrix([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
v = Matrix([[0.0], [0.0], [0.0], [0.0], [0.0]])
V = 0.0
n = 0

###############################################################
### DATA GENERATION and CANONICAL INFORMATION UPDATING ###
x = np.zeros(nn)
y = np.zeros(nn)
var_f_est = np.zeros(nn)
var_est_f_est = np.zeros(nn)

for j in range(nn):
	### generate new point
	(x[j], y[j]) = gen(a, av, sigma, x_left, x_right)
	### elementary information calculation
	Ti = (F.T).subs(z, x[j])*F.subs(z, x[j])
	vi = (F.T).subs(z, x[j])*y[j]
	Vi = y[j]**2
	ni = 1
	### canonical information updating
	T += Ti
	v += vi
	V += Vi
	n += ni
	

print(T.det())
### f(x) estimation
f_est = F*T.inv()*v
### var[f(x)] estimation, if sigma is known
var_f_est = sigma**2*(F*T.inv())*F.T
### var[f(x)] estimation, if sigma is unknown
var_est_f_est = ( Matrix([[V]]) - (v.T)*T.inv()*v )/(n-m) * F*T.inv()*(F.T)

###############################################################
### GRAPHS ###
### some additional manipulation with data format for graph plotting (converting from SymPy to NumPy)
x_dots = np.arange(x_left, x_right, step)
f_est_num = np.zeros(x_dots.shape[0])
var_f_est_sub = np.zeros(x_dots.shape[0])
var_f_est_super = np.zeros(x_dots.shape[0])
var_est_f_est_sub = np.zeros(x_dots.shape[0])
var_est_f_est_super = np.zeros(x_dots.shape[0])

for j in range(x_dots.shape[0]):
	f_est_num[j] = np.asscalar(np.array(f_est.subs(z, x_dots[j]).evalf()).astype(np.float64))
	var_f_est_sub[j] = f_est_num[j] - np.sqrt(np.asscalar(np.array(var_f_est.subs(z, x_dots[j]).evalf()).astype(np.float64)))
	var_f_est_super[j] = f_est_num[j] + np.sqrt(np.asscalar(np.array(var_f_est.subs(z, x_dots[j]).evalf()).astype(np.float64)))
	var_est_f_est_sub[j] = f_est_num[j] - np.sqrt(np.asscalar(np.array(var_est_f_est.subs(z, x_dots[j]).evalf()).astype(np.float64)))
	var_est_f_est_super[j] = f_est_num[j] + np.sqrt(np.asscalar(np.array(var_est_f_est.subs(z, x_dots[j]).evalf()).astype(np.float64)))

### grap plotting
x_dots = np.arange(x_left, x_right, step)
plt.rcParams["figure.figsize"] = (10,7)
plt.rc('font', size=18) 
plt.plot(x, y, 'ro', label='y(x)')
plt.plot(x_dots, f(a, x_dots), 'b-', label='f(x)')
plt.plot(x_dots, f_est_num, 'g-', label='f_est(x)')
plt.plot(x_dots, var_f_est_sub, 'g--', label='sigma is known')
plt.plot(x_dots, var_f_est_super, 'g--', label='')
plt.plot(x_dots, var_est_f_est_sub, 'm--', label='sigma is unknown')
plt.plot(x_dots, var_est_f_est_super, 'm--', label='')
plt.legend()
plt.grid(True)
plt.savefig('f.png')
plt.close()
