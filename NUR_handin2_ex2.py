#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:36:13 2023

@author: stroo
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z,Tc,psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k


def equilibrium2(T,Z,Tc,psi, nH, A, xi):
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)

def falseposition(func,params, bracket, target):
    start = time.perf_counter()
    a = bracket[0]
    b = bracket[1]
    finished = False
    steps=0
    while finished != True:
        #Find a point between the brackets using linear interpolation
        c = b + ((b-a)/(func(a,*params)-func(b,*params))*func(b,*params))
        steps+=1
        if np.abs(func(c,*params)) < target:
            finished = True
            stop = time.perf_counter()
            print('The number of steps taken to find the root is {} steps and the time taken is {} seconds'.format(steps, stop-start))
            return c
        #Make sure the points we choose to continue with bracket the root
        if func(a,*params)*func(c,*params) < 0:
            b = c
        else:
            a = c
    
k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s   
parameters1 = [0.015, 10**4, 0.929]
root1 = (falseposition(equilibrium1, parameters1, [1,10**7],10**-20))
x1=np.linspace(1,10**7,1000)
plt.plot(x1, equilibrium1(x1, *parameters1))
plt.plot(root1, equilibrium1(root1,*parameters1), 'or')
plt.xscale('log')
plt.grid()
plt.title('Simplified Heating/Cooling equilibrium')
plt.xlabel('Temperature T (K)')
plt.ylabel('Difference between heating and cooling')
plt.savefig('./plot/root1.pdf')
plt.show()

print('In the simplified model, the equilibrium temperature is ', root1)
n_e = [10**-4,1, 10**4]
for i  in range(len(n_e)):
    parameters2 = [0.015,10**4,0.929,n_e[i], 5*10**-10, 10**-15]
    root2 = falseposition(equilibrium2,parameters2,[1,10**15], np.abs(equilibrium2(1,*parameters2))*10**-1)
    x2=np.linspace(1,10**15,1000)
    plt.plot(x2,equilibrium2(x2,*parameters2))
    plt.grid()
    plt.plot(root2,equilibrium2(root2,*parameters2),'or')
    plt.xscale('log')
    plt.title(r'Heating/Cooling equilibrium for n_e = {} cm^-3'.format(n_e[i]))
    plt.xlabel('Temperature T (K)')
    plt.ylabel("Difference between heating and cooling")
    plt.savefig('./plot/root{}.pdf'.format(i+2))
    plt.show()
    print('With a density of ', n_e[i], ', the equilibrium temperature is', root2)
