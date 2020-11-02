#!/usr/bin/env python
# coding: utf-8

# Import the necessary modules/libraries
import numpy as np
from numpy import poly1d
import math
from random import random
import itertools

# Convert a non-zero vector in C2 into a degree less than or equal to 1 polynomial of one variable t
def poly(u):
    if u.size!=2:
        print("Error, the vector is not in C2")
        return
    if np.linalg.norm(u)==0:
        print("Error, the vector in C2 is 0")
        return
    return poly1d(np.array([u[0],-u[1]]), variable="t")

# Compute the 2 by 2 determinant of two vectors in u and v in C2
def det(u,v):
    mytemp = np.zeros((2,2),dtype=np.complex)
    mytemp[0,:] = u[:]
    mytemp[1,:] = v[:]
    return np.linalg.det(mytemp)

# Compute the null direction from xa to xb
def null_direction(xa, xb):
    if (xa.shape != (2,2)) or (xb.shape != (2,2)):
        print("Error: xa and xb must both be 2 by 2 arrays.")
        return
    cab = 0.5*(xa+xb)
    dab = 0.5*(xb-xa)
    dab_mod = (np.linalg.inv(cab)).dot(dab)
    factor = - (1./(2.*np.linalg.det(dab_mod)))
    la = factor*(np.trace(dab_mod) + np.sqrt((np.trace(dab_mod))**2-4.*                            np.linalg.det(dab_mod)))
    vab = cab + la*dab
    return (2./np.trace(vab))*vab

# Compute all null directions from xa to xb, for a different from b
def all_null_directions(conf):
    n = conf.shape[0]
    all_null_dirs = np.zeros((n,n,2), dtype = np.complex)
    for a in range(n):
        for b in range(n):
            if a != b:
                null_dir = null_direction(conf[a],conf[b])
                first_col = null_dir[:,0]
                second_col = null_dir[:,1]
                norm_first_col = np.linalg.norm(first_col)
                if norm_first_col > 1.e-8:
                    all_null_dirs[a,b,:] = first_col/norm_first_col
                else:
                    norm_second_col = np.linalg.norm(second_col)
                    all_null_dirs[a,b,:] = second_col/norm_second_col
    return all_null_dirs

# generate N "random" configurations of n points
def generate_random_confs(n,N):
    confs = []
    for l in range(N):
        conf = []
        for a in range(n):
            x = np.eye(2) + (random()-0.5)*np.array([[0.,1.],[1.,0.]]) +                (random()-0.5)*np.array([[0.,-1.j],[1.j,0.]]) +                (random()-0.5)*np.array([[1.,0.],[0.,-1.]])
            conf.append(x)
        confs.append(conf)
    return np.array(confs)

# compute the AS hyperbolic determinant of a configuration
def AS_det(conf):
    eps = 1.e-10
    n = conf.shape[0]
    p = all_null_directions(conf)

    M=np.zeros((n,n),dtype=np.complex)

    for i in range(n):
        mypoly = poly1d([complex(1,0)],variable="t")
        for j in range(n):
            if (j!=i):
                mypoly=mypoly*poly(p[i,j,:])
        mycoeffs = mypoly.coeffs[::-1]
        M[i,0:mycoeffs.size]=mycoeffs[:] 
    
    P = 1.
    for i in range(n-1):
        for j in range(i+1,n):
            P*=det(p[i,j],p[j,i])

    return np.linalg.det(M)*P**(-1)

# Below are some lines to demonstrate how to call the above functions

# confs = generate_random_confs(4,100)
# dets = np.array([AS_det(conf) for conf in confs])
# min(np.real(dets)), max(np.real(dets))
# min(abs(dets)), max(abs(dets))