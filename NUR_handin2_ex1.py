# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:00:15 2023

@author: Jelme
"""
import numpy as np
import matplotlib.pyplot as plt

def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/(b))**(a-3))*np.exp(-(x/b)**c)

def n_spherical(x, A, Nsat,a,b,c):
    return 4*np.pi*A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)*x**2

def romberg(left, right, m, func, params):
    """
    Numerical integration using the Romberg method.

    Parameters
    ----------
    left : int or float
        The left bound of the integration.
    right : int or float
        The right bound of the integration.
    m : int
        The order or number of points to be created.
    func : function
        The function to be integrated.
    params : list
        Any additional parameters the function requires.

    Returns
    -------
    float
        The resukt of the numerical integration.

    """
    h = right - left
    r = np.zeros(m)
    r[0] = 0.5 * h * (func(left,*params)+func(right,*params))
    N_p = 1
    #Create m initial approximations using trapezoid approximation
    for i in range(1,m):
        r[i] = 0
        delta = h
        h = 0.5 * h
        x = left + h
        for j in range(N_p):
            r[i] = r[i] + func(x,*params)
            x = x + delta
        r[i] = 0.5 * (r[i-1] + delta*r[i])
        N_p *= 2

    N_p = 1
    #Combine the approximations in an analogue to Neville's algorithm
    for i in range(1,m):
        N_p *= 4
        recip = 1/(N_p-1) #Calculates the denominator for the next loop for a little extra efficiency
        for j in range(m-i):
            r[j] = (N_p*r[j+1] - r[j])*recip
    return r[0]

def rejection_sampling(p, N, params,xrange=1,yrange=1):
    """
    Gather N samples from a given distribution p using the Rejection method.

    Parameters
    ----------
    p : func
        The distribution samples should be taken from.
    N : int
        The number of samples to be taken.
    params : list
        Any additional parameters the function requires.

    Returns
    -------
    sample : list
        A random sample from the distribution.

    """
    counter = 0
    sample = np.zeros(N)
    #seeds were created by mashing a hand on the numpad :)
    seed_x=8641843
    seed_y=1546416
    while counter < N:
        #The RNG generates values between 0 and 1, x and y should be scaled to 
        #the distribution's range
        x = xrange*RNG(seed_x,1)
        y = yrange*RNG(seed_y,1)
        p_x = p(x, *params)
        if y <= p_x:
            sample[counter] = x
            counter += 1
            plt.plot(x,y,',')
        #Return x and y to a usable form for a seed by multiplying by the modulus
        #Only works for this specific RNG generator, but I did not have time to
        #make a more general implementation.
        seed_x = x/xrange * 2**32
        seed_y = y/yrange * 2**32
    plt.show()
    return sample

def selection_sort(data, sortcol=None):
    """
    Sort an array in-place from low to high values using the selection sort algorithm.

    Parameters
    ----------
    data : np.array
        The array to be sorted.
    sortcol : int, optional
        The column to be sorted. The other values in the column will be swapped
        accordingly. The default is None, if the array is 1-dimensional.

    Returns
    -------
    data : array
        The sorted array.

    """
    for i in range(len(data[:,sortcol])-1):
        imin = i
        for j in range(i,len(data)):
            if data[j,sortcol] < data[imin,sortcol]:
                imin = j
        if imin != i:
            data[[i,imin]] = data[[imin,i]]
    return data

def RNG(seed, nr, a=1664525, c=1013904223, m=2**32):
    """
    Random number generation using a linear congruential generator.
    The default values for the parameters are taken from the Numerical Recipes book.
    As long as nr < m, these values should generate a list of non-repeating 
    pseudo-random numbers between 0 and 1.

    Parameters
    ----------
    seed : int
        The initial seed for the RNG.
    nr : int
        The number of random values to be generated.
    a : int, optional
        The a value in the LCG. The default is 1664525.
    c : int, optional
        The c value in the LCG. The default is 1013904223.
    m : int, optional
        The modulo in the LCG. The default is 2**32.

    Returns
    -------
    sequence : int or array
        If nr = 1, returns the number generated, otherwise returns an array of random values.

    """
    sequence = np.zeros(nr)
    start = int((a*seed+c)%m)
    #Combination with 64-bit XORshift was planned, but not implemented
    # start ^= (start>>21)
    # start^=(start<<35)
    # start^=(start>>4)
    if nr != 1:
        sequence[0] = start
        for i in range(1,nr):
            rand = int((a*sequence[i-1]+c)%m)
            # rand^=(rand>>21)
            # rand^=(rand<<35)
            # rand^=(rand>>4)
            sequence[i] = rand
    else:
        sequence=start
    sequence /= m
    return sequence

def ridder(func, params, x, target, m, h):
    """
    Numerical differentiation using Ridder's method.

    Parameters
    ----------
    func : function
        The function to be differentiated.
    params : list
        Any additional parameters the function requires.
    x : int or float
        The value where to differentiate the function.
    target : float
        The target error below which to stop iterating.
    m : int
        The initial order of approximation. Will be increased if necessary, but
        that will increase run time.
    h : int
        The initial range of approximation.

    Returns
    -------
    float
        Result of the numerical differentiation.

    """
    approx = np.zeros(m)
    d = 0.5
    last_approx = 0
    last_improv = 100000
    #Create a list of initial approximations
    for i in range(m):
        approx[i] = (func(x+h, *params) - func(x-h, *params))/(2*h)
        h *= d
    #combine the initial approximations to reduce the error analoguously to Neville
    for j in range(1,m):
        for i in range(m-j):
          approx[i] =  (d**(2*(j+1))*approx[i+1]-approx[i])/(d**(2*(j+1))-1)
          if approx[i] - last_approx > last_improv:
              print('Error grew, return best approximation')
              return last_approx
          elif np.abs((approx[i] - last_approx) - last_improv) < target:
              print('Target met, return best approximation')
              return approx[i]
          else:
              
              last_improv=approx[i] - last_approx
              last_approx=approx[i]
    m += 1
    #If neither the error grew nor the target was met, run the algorithm again
    #but this time with an additional initial approximation
    print('Try again with higher m: ', m)
    ridder(func,params,x,target,m,h)
    
A=1. # to be computed
Nsat=100
a=2.4
b=0.25
c=1.6
parameters=[A,Nsat,a,b,c]
integrand = romberg(10**-20, 5, 10, n_spherical, parameters)

A = 100/integrand
parameters[0] = A
print('The value of A is: ',A)


points = rejection_sampling(n_spherical, 10000, parameters,5,n_spherical(10**-4,*parameters))
# binedges = np.logspace(-4,np.log10(5),num=20,endpoint=False)

# counts, bins= np.histogram(points, binedges)
# plt.stairs(counts,bins,fill=True)
# plt.plot(np.linspace(10**-4,5,10000),n_spherical(np.linspace(10**-4,5,10000),*parameters))
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Comparison of the distribution and samples from the distribution')
# plt.xlabel('p(x)')
# plt.ylabel('x')
# plt.show()

slope = (ridder(n,parameters, 1,10**-4,6,0.01))
print('The value of dn/dx at x=1 is: ',slope)



#Link a randomly generated number to a list of indexes, sort the indexes
#based on their random numbers, and then take the lowest 100:
#If the RNG is unbiased, this will uniquely sample 100 points from the larger 
#sample, with each point having the same probibility of being picked.
index = np.vstack((np.arange(10000,dtype=int),RNG(354876464,10000))).T
index = np.int32(selection_sort(index,sortcol=1)[:100,0])
sample = points[index]
sample= selection_sort(sample)

xrange = np.logspace(-4,np.log10(5),num=1000)
cum = np.zeros(len(xrange))
for i in range(len(xrange)):
    #The cumulative is the amount of points within a certain radius
    cum[i] = (sample < xrange[i]).sum()

plt.plot(xrange,cum)
plt.title('Cumulative distribution function')
plt.xlabel('Number of galaxies')
plt.ylabel('Radius x')
plt.xscale('log')
plt.show()

